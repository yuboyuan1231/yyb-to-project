from __future__ import annotations

import ctypes
import fcntl
import json
import os
import re
import shutil
import stat
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .atomic_io import (
    BootstrapLease,
    WriterLease,
    append_recoverable_jsonl,
    assert_allowed_write_path,
    atomic_write_bytes,
    atomic_write_immutable,
    fsync_directory,
    load_json,
)
from .canonical import (
    business_inventory,
    bytes_sha256,
    canonical_json_bytes,
    lstat_identity,
    read_regular_bytes,
    regular_file_record,
    semantic_sha256,
)
from .constants import (
    ALLOWED_WRITE_GLOBS,
    BLUEPRINT,
    BOOTSTRAP_LOCK,
    BOOTSTRAP_LOCK_HEADER,
    BOOTSTRAP_FIXTURES,
    CACHE_ROOT,
    CALIB_HOLDOUT_MANIFEST,
    CALIB_SELECT_MANIFEST,
    CANONICAL_ROOTS,
    CHECKPOINT,
    CHECKPOINT_ROOT,
    CODE_ROOT,
    CONTROL_ROOT,
    DATA_ROOT,
    ENTRYPOINT,
    EXPECTED_BLUEPRINT_SHA256,
    EXPECTED_BRANCH,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CHECKPOINT_SIZE,
    EXPECTED_GOAL_CONTRACT_SHA256,
    EXPECTED_HANDOFF_SHA256,
    EXPECTED_HEAD,
    FEATURE_SOURCES,
    G0_REVIEWED_FILES,
    GOAL_CONTRACT,
    GOAL_ID,
    GOAL_OBJECTIVE,
    GOAL_OWNED_EXACT_REPO_PATHS,
    GOAL_OWNED_REPO_PREFIXES,
    HANDOFF,
    HISTORICAL_ARTIFACTS,
    HISTORICAL_DENY_WRITE_ROOTS,
    PRE_G0_SECURITY_INCIDENT,
    PRE_G0_SECURITY_INCIDENT_IMPACT_ASSESSMENT,
    PRE_G0_SECURITY_INCIDENT_USER_ACK,
    PRE_G0_SECURITY_INCIDENT_USER_ACK_TEXT,
    PROTECTED_HISTORICAL_C28C_RESULTS,
    PROTECTED_VAL_JSONL,
    PSEUDO_OFFICIAL_MANIFEST,
    QUARANTINED_G0_ATTEMPTS,
    QUARANTINE_ROOT,
    REPORT_ROOT,
    REPO_ROOT,
    SCHEMA_VERSION,
    STATIC_REVIEW_RECEIPT,
    TEST_ROOT,
    TRAIN_CORPUS_AUDITS,
    TRAIN_CORPUS_COUNT,
    TRAIN_CORPUS_VIDEO_ID_ORDER_SHA256,
    TRAIN_FIT_MANIFEST,
    TRAIN_JSONL,
    VIDEO_META,
)


class PreflightError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_result(argv: list[str], *, cwd: Path = REPO_ROOT, timeout: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return {
            "argv_sha256": bytes_sha256(canonical_json_bytes(argv)),
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr_sha256": bytes_sha256(proc.stderr),
            "timed_out": False,
            "spawn_error": None,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "argv_sha256": bytes_sha256(canonical_json_bytes(argv)),
            "returncode": None,
            "stdout": exc.stdout or b"",
            "stderr_sha256": bytes_sha256(exc.stderr or b""),
            "timed_out": True,
            "spawn_error": None,
        }
    except OSError as exc:
        error_fingerprint = f"{type(exc).__name__}:{getattr(exc, 'errno', None)}".encode("utf-8")
        return {
            "argv_sha256": bytes_sha256(canonical_json_bytes(argv)),
            "returncode": None,
            "stdout": b"",
            "stderr_sha256": bytes_sha256(error_fingerprint),
            "timed_out": False,
            "spawn_error": type(exc).__name__,
        }


def _run_checked(argv: list[str], *, cwd: Path = REPO_ROOT) -> bytes:
    result = _run_result(argv, cwd=cwd)
    if result["timed_out"] or result["returncode"] != 0:
        raise PreflightError(
            f"command failed: argv_sha={result['argv_sha256']} rc={result['returncode']} timeout={result['timed_out']}"
        )
    return bytes(result["stdout"])


def _is_goal_owned_repo_path(path: str) -> bool:
    normalized = Path(path).as_posix()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized in GOAL_OWNED_EXACT_REPO_PATHS or any(
        normalized.startswith(prefix) for prefix in GOAL_OWNED_REPO_PREFIXES
    )


def _parse_porcelain_z(raw: bytes) -> list[dict[str, str]]:
    parts = raw.split(b"\0")
    entries: list[dict[str, str]] = []
    index = 0
    while index < len(parts):
        token = parts[index]
        index += 1
        if not token:
            continue
        text = token.decode("utf-8", "surrogateescape")
        if len(text) < 4 or text[2] != " ":
            raise PreflightError(f"invalid git porcelain entry: {text!r}")
        status_code = text[:2]
        entry = {"status": status_code, "destination_path": text[3:]}
        if "R" in status_code or "C" in status_code:
            if index >= len(parts) or not parts[index]:
                raise PreflightError("truncated rename/copy porcelain record")
            entry["source_path"] = parts[index].decode("utf-8", "surrogateescape")
            index += 1
        entries.append(entry)
    return entries


def _metadata_for_repo_path(relative: str) -> dict[str, Any]:
    path = REPO_ROOT / relative
    if not os.path.lexists(path):
        return {"path": relative, "exists": False}
    record = lstat_identity(path)
    record["path"] = relative
    record["exists"] = True
    return record


def _protected_git_exclusion_pathspecs(
    protected_records: list[dict[str, Any]],
) -> list[str]:
    candidates: list[Path] = []
    for record in protected_records:
        declared = record.get("declared") or record.get("path")
        if not declared:
            continue
        path = Path(str(declared))
        try:
            relative = path.relative_to(REPO_ROOT)
        except ValueError:
            continue
        candidates.append(relative)
    minimal: list[Path] = []
    for relative in sorted(set(candidates), key=lambda item: (len(item.parts), item.as_posix())):
        if any(parent == relative or parent in relative.parents for parent in minimal):
            continue
        minimal.append(relative)
    return [f":(exclude,top,literal){relative.as_posix()}" for relative in minimal]


def collect_git_inventory(protected_records: list[dict[str, Any]]) -> dict[str, Any]:
    protected_exclusions = _protected_git_exclusion_pathspecs(protected_records)
    branch = _run_checked(["git", "branch", "--show-current"]).decode().strip()
    head = _run_checked(["git", "rev-parse", "HEAD"]).decode().strip()
    porcelain_raw = _run_checked(
        ["git", "status", "--porcelain=v1", "-z", "-uall", "--", ".", *protected_exclusions]
    )
    porcelain = _parse_porcelain_z(porcelain_raw)
    tracked_diff = _run_checked(
        ["git", "diff", "--raw", "--no-abbrev", "-z", "--", ".", *protected_exclusions]
    )
    staged_diff = _run_checked(
        ["git", "diff", "--cached", "--raw", "--no-abbrev", "-z", "--", ".", *protected_exclusions]
    )
    untracked_raw = _run_checked(
        ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", ".", *protected_exclusions]
    )
    ignored_result = _run_result(
        [
            "git",
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
            "--",
            ".",
            *protected_exclusions,
        ]
    )
    if ignored_result["timed_out"] or ignored_result["returncode"] != 0:
        raise PreflightError("unable to inventory ignored paths")
    ignored_raw = bytes(ignored_result["stdout"])
    untracked = sorted(
        item.decode("utf-8", "surrogateescape") for item in untracked_raw.split(b"\0") if item
    )
    ignored = sorted(
        item.decode("utf-8", "surrogateescape") for item in ignored_raw.split(b"\0") if item
    )

    external_status_endpoints: list[dict[str, Any]] = []
    cross_boundary_records: list[dict[str, Any]] = []
    for entry in porcelain:
        endpoint_flags: list[tuple[str, str, bool]] = []
        for role in ("source_path", "destination_path"):
            if role in entry:
                endpoint_flags.append((role, entry[role], _is_goal_owned_repo_path(entry[role])))
        if len({owned for _role, _path, owned in endpoint_flags}) > 1:
            cross_boundary_records.append(entry)
        for role, relative, owned in endpoint_flags:
            if not owned:
                external_status_endpoints.append(
                    {
                        "status": entry["status"],
                        "role": role,
                        "identity": _metadata_for_repo_path(relative),
                    }
                )

    external_untracked = [path for path in untracked if not _is_goal_owned_repo_path(path)]
    external_ignored = [path for path in ignored if not _is_goal_owned_repo_path(path)]
    external_path_metadata = [
        _metadata_for_repo_path(path) for path in sorted(set(external_untracked + external_ignored))
    ]
    preexisting = {
        "schema_version": "c28f_preexisting_dirty_inventory_v2",
        "external_status_endpoints": sorted(
            external_status_endpoints,
            key=lambda item: (
                item["identity"]["path"],
                item["role"],
                item["status"],
            ),
        ),
        "external_untracked_paths": external_untracked,
        "external_ignored_paths": external_ignored,
        "external_path_metadata": external_path_metadata,
        "tracked_diff_sha256": bytes_sha256(tracked_diff),
        "staged_diff_sha256": bytes_sha256(staged_diff),
        "cross_goal_external_rename_or_copy": cross_boundary_records,
    }
    if cross_boundary_records:
        raise PreflightError("git rename/copy crosses Goal-owned and external path domains")
    return {
        "branch": branch,
        "head": head,
        "porcelain_sha256": bytes_sha256(porcelain_raw),
        "tracked_diff_sha256": bytes_sha256(tracked_diff),
        "staged_diff_sha256": bytes_sha256(staged_diff),
        "preexisting_dirty_inventory": preexisting,
        "preexisting_dirty_inventory_sha256": bytes_sha256(canonical_json_bytes(preexisting)),
        "all_untracked_path_count": len(untracked),
        "all_untracked_paths_sha256": bytes_sha256(canonical_json_bytes(untracked)),
        "all_ignored_path_count": len(ignored),
        "all_ignored_paths_sha256": bytes_sha256(canonical_json_bytes(ignored)),
        "goal_owned_status_entries": [
            entry
            for entry in porcelain
            if _is_goal_owned_repo_path(entry["destination_path"])
            or _is_goal_owned_repo_path(entry.get("source_path", ""))
        ],
        "protected_worktree_pathspec_exclusions": protected_exclusions,
        "protected_worktree_content_open_policy": "excluded from git status/diff; metadata locked separately",
    }


def _assert_path_chain_no_symlink(path: Path) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        exists = os.path.lexists(current)
        record: dict[str, Any] = {"path": str(current), "exists": exists}
        if exists:
            identity = lstat_identity(current)
            record.update(_stable_identity_fields(identity))
            if identity["type"] == "symlink":
                raise PreflightError(f"symlink component is forbidden: {current}")
        chain.append(record)
    return chain


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _stable_identity_fields(record: dict[str, Any]) -> dict[str, Any]:
    """Keep path identity, not mutable directory timestamps or entry-count-derived size."""
    keys = ["path", "resolved", "type", "device", "inode", "mode"]
    if record.get("type") != "directory":
        keys.append("nlink")
    return {key: record[key] for key in keys if key in record}


def _stable_metadata_or_missing(path: Path) -> dict[str, Any]:
    if os.path.lexists(path):
        identity = lstat_identity(path)
        return {"declared": str(path), "exists": True, **_stable_identity_fields(identity)}
    parent = path.parent
    while not os.path.lexists(parent):
        parent = parent.parent
    return {
        "declared": str(path),
        "exists": False,
        "resolved": str(path.resolve(strict=False)),
        "nearest_existing_parent": _stable_identity_fields(lstat_identity(parent)),
    }


def _metadata_or_missing(path: Path) -> dict[str, Any]:
    if os.path.lexists(path):
        return {"declared": str(path), "exists": True, **lstat_identity(path)}
    parent = path.parent
    while not os.path.lexists(parent):
        parent = parent.parent
    return {
        "declared": str(path),
        "exists": False,
        "resolved": str(path.resolve(strict=False)),
        "nearest_existing_parent": _stable_identity_fields(lstat_identity(parent)),
    }


def discover_protected_paths() -> list[Path]:
    protected: set[Path] = {
        PROTECTED_VAL_JSONL,
        PROTECTED_HISTORICAL_C28C_RESULTS,
        CALIB_HOLDOUT_MANIFEST,
        PSEUDO_OFFICIAL_MANIFEST,
        DATA_ROOT / "data/tvr_val_top100_hero",
        DATA_ROOT / "data/tvr_test_public_top100_hero",
        DATA_ROOT / "data/tvr_test_public_release.jsonl",
    }
    tokens = ("official", "holdout", "tvr_val", "tvr_test")
    for current, dirnames, filenames in os.walk(REPO_ROOT, topdown=True, followlinks=False):
        base = Path(current)
        keep: list[str] = []
        for name in dirnames:
            candidate = base / name
            lowered = name.lower()
            if any(token in lowered for token in tokens):
                protected.add(candidate)
                continue
            if candidate.is_symlink():
                continue
            keep.append(name)
        dirnames[:] = keep
        for name in filenames:
            if any(token in name.lower() for token in tokens):
                protected.add(base / name)
    data_dir = DATA_ROOT / "data"
    if data_dir.exists():
        with os.scandir(data_dir) as iterator:
            for entry in iterator:
                if any(token in entry.name.lower() for token in tokens):
                    protected.add(Path(entry.path))
    return sorted(protected, key=lambda item: str(item))


def collect_protected_metadata_records() -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for root in discover_protected_paths():
        _assert_path_chain_no_symlink(root)
        root_record = _metadata_or_missing(root)
        if root_record.get("exists") and root_record.get("type") == "symlink":
            raise PreflightError(f"protected registry contains an ambiguous symlink: {root}")
        by_path[str(root)] = root_record
        if not root_record.get("exists") or root_record.get("type") != "directory":
            continue
        for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            base = Path(current)
            kept: list[str] = []
            for name in dirnames:
                path = base / name
                record = _metadata_or_missing(path)
                if record.get("type") == "symlink":
                    raise PreflightError(f"protected registry contains an ambiguous symlink: {path}")
                by_path[str(path)] = record
                if record.get("type") != "symlink":
                    kept.append(name)
            dirnames[:] = kept
            for name in filenames:
                path = base / name
                record = _metadata_or_missing(path)
                if record.get("type") == "symlink":
                    raise PreflightError(f"protected registry contains an ambiguous symlink: {path}")
                by_path[str(path)] = record
    return [by_path[key] for key in sorted(by_path)]


def _assert_control_namespace_not_protected(protected_records: list[dict[str, Any]]) -> None:
    forbidden = _identity_pairs(protected_records)
    for path, object_type in _walk_namespace(REPORT_ROOT):
        identity = lstat_identity(path)
        if object_type == "file" and (
            int(identity["device"]), int(identity["inode"])
        ) in forbidden:
            raise PreflightError(f"control file aliases a protected inode before open: {path}")


def _assert_content_source_not_protected(
    path: Path,
    protected_records: list[dict[str, Any]],
) -> None:
    metadata = _metadata_or_missing(path)
    if not metadata.get("exists") or metadata.get("type") != "file":
        raise PreflightError(f"content source is not a regular file: {path}")
    identity = (int(metadata["device"]), int(metadata["inode"]))
    resolved = Path(metadata["resolved"])
    for protected in protected_records:
        protected_resolved = Path(str(protected.get("resolved", protected.get("declared", ""))))
        if protected.get("exists") and "device" in protected and identity == (
            int(protected["device"]),
            int(protected["inode"]),
        ):
            raise PreflightError(f"content source aliases a protected inode before open: {path}")
        if protected_resolved and (
            _contains(protected_resolved, resolved) or _contains(resolved, protected_resolved)
        ):
            raise PreflightError(f"content source overlaps a protected path before open: {path}")


def _manifest_record(
    path: Path,
    *,
    expected_count: int,
    protected_records: list[dict[str, Any]],
) -> dict[str, Any]:
    _assert_content_source_not_protected(path, protected_records)
    data, record = read_regular_bytes(path, max_bytes=16 * 1024 * 1024)
    try:
        lines = [line.strip() for line in data.decode("utf-8").splitlines() if line.strip()]
        ids = [int(line) for line in lines]
    except (UnicodeDecodeError, ValueError) as exc:
        raise PreflightError(f"invalid desc-ID manifest: {path}") from exc
    if len(ids) != expected_count or len(set(ids)) != len(ids):
        raise PreflightError(
            f"desc-ID manifest completeness mismatch: {path}: count={len(ids)} unique={len(set(ids))}"
        )
    return {**record, "line_count": len(ids), "entity_type": "desc_id"}


def _protected_manifest_declared_record(
    path: Path,
    *,
    expected_count: int,
    expected_sha256: str,
) -> dict[str, Any]:
    metadata = _metadata_or_missing(path)
    if not metadata.get("exists") or metadata.get("type") != "file":
        raise PreflightError(f"protected desc-ID manifest is missing: {path}")
    return {
        **metadata,
        "entity_type": "desc_id",
        "declared_line_count": expected_count,
        "expected_sha256": expected_sha256,
        "sha256": None,
        "verification_status": "PRE_REGISTERED_HASH_METADATA_ONLY_NOT_OPENED_IN_G0",
        "content_opened": False,
        "content_hash_read": False,
    }


def collect_corpus_and_splits(protected_records: list[dict[str, Any]]) -> dict[str, Any]:
    splits = {
        "train_fit": _manifest_record(
            TRAIN_FIT_MANIFEST,
            expected_count=69_428,
            protected_records=protected_records,
        ),
        "calib_select": _manifest_record(
            CALIB_SELECT_MANIFEST,
            expected_count=8_677,
            protected_records=protected_records,
        ),
        "calib_holdout_desc_id_only": _protected_manifest_declared_record(
            CALIB_HOLDOUT_MANIFEST,
            expected_count=4_340,
            expected_sha256="7d5dbe30a18ef6e6320a63521e872c362689a163c937af1d43038e73d34c2afa",
        ),
        "pseudo_official_desc_id_only": _protected_manifest_declared_record(
            PSEUDO_OFFICIAL_MANIFEST,
            expected_count=4_339,
            expected_sha256="d250205897b827c964d0a59326dc2d63163c528d983aec1c3f955823fd17fc14",
        ),
    }
    expected = {
        "train_fit": "24007d3f6f1159f55e68a8ee64c779436a34db4bff601f9d9cd74ee6e102bf0b",
        "calib_select": "b2c2a6ec3386d1ff7a3ffd409a8b0f3859ddfb7edddab34d607981b0345a467d",
    }
    for name, digest in expected.items():
        if splits[name]["sha256"] != digest:
            raise PreflightError(f"split hash mismatch: {name}")
    corpus_audits: dict[str, Any] = {}
    corpus_claim_assertions: dict[str, Any] = {}
    for name, specification in TRAIN_CORPUS_AUDITS.items():
        _assert_content_source_not_protected(specification["path"], protected_records)
        audit_bytes, record = read_regular_bytes(specification["path"], max_bytes=8 * 1024 * 1024)
        if record["sha256"] != specification["sha256"]:
            raise PreflightError(f"train-only corpus audit hash mismatch: {name}")
        corpus_audits[name] = record
        if name == "c12_corpus_index_audit":
            payload = json.loads(audit_bytes.decode("utf-8"))
            if (
                payload.get("video_count") != TRAIN_CORPUS_COUNT
                or payload.get("video_to_pos_unique") is not True
                or payload.get("no_duplicate_overwrite") is not True
                or payload.get("position_based_join") is not False
            ):
                raise PreflightError("machine-readable train corpus count/index claim mismatch")
            corpus_claim_assertions[name] = {
                "source_key": "video_count",
                "asserted_value": TRAIN_CORPUS_COUNT,
                "unique_index": True,
                "position_based_join": False,
            }
        elif name == "c12_feature_inventory":
            marker = b'"train_video_count": 17435'
            subtitle_marker = b'"subtitle_video_missing_full": 0'
            visual_marker = b'"visual_video_missing_full": 0'
            if (
                audit_bytes.count(marker) != 1
                or audit_bytes.count(subtitle_marker) != 1
                or audit_bytes.count(visual_marker) != 1
            ):
                raise PreflightError("train feature inventory corpus/key claim is missing or ambiguous")
            corpus_claim_assertions[name] = {
                "source_key": "full_key_existence.train_video_count",
                "asserted_value": TRAIN_CORPUS_COUNT,
                "missing_subtitle_video_keys": 0,
                "missing_visual_video_keys": 0,
            }
    feature_sources = {name: _metadata_or_missing(path) for name, path in FEATURE_SOURCES.items()}
    if not all(record["exists"] for record in feature_sources.values()):
        raise PreflightError("required feature source is missing")
    feature_manifest = {
        "corpus_video_count": TRAIN_CORPUS_COUNT,
        "corpus_video_id_sha256": TRAIN_CORPUS_VIDEO_ID_ORDER_SHA256,
        "train_only_corpus_audits": corpus_audits,
        "corpus_identity_provenance": {
            "count_claim_assertions": corpus_claim_assertions,
            "ordered_video_id_sha256": TRAIN_CORPUS_VIDEO_ID_ORDER_SHA256,
            "ordered_video_id_hash_source": (
                "PRE_REGISTERED_G0_CODE_CONSTANT_FROM_PRIOR_NONPROTECTED_TRAIN_CORPUS_AUDIT"
            ),
            "corroborating_source_audit_sha256": {
                name: specification["sha256"] for name, specification in TRAIN_CORPUS_AUDITS.items()
            },
            "hash_derivation": "sha256(newline_join(ordered_train_video_ids)+newline)",
            "g1_requirement": (
                "enumerate train-only feature-store keys, reproduce count and ordered-ID SHA, and persist "
                "a standalone corpus manifest before any model forward"
            ),
        },
        "source_file_identities": feature_sources,
        "scope": (
            "pre-registered train-only corpus identity plus source metadata only; enumerate only "
            "feature-store keys behind the G1 firewall and require a derived-bank full SHA before F0-A"
        ),
    }
    return {
        "video_count": TRAIN_CORPUS_COUNT,
        "video_id_order_sha256": TRAIN_CORPUS_VIDEO_ID_ORDER_SHA256,
        "video_metadata_mixed_container": {
            **_metadata_or_missing(VIDEO_META),
            "content_opened": False,
            "content_hash_read": False,
            "reason": "mixed train/val/test duration-index container is forbidden in G0",
        },
        "feature_manifest": feature_manifest,
        "feature_manifest_sha256": bytes_sha256(canonical_json_bytes(feature_manifest)),
        "mixed_train_record_source": {
            **_metadata_or_missing(TRAIN_JSONL),
            "content_opened": False,
            "reason": "mixed query-role JSONL is forbidden in G0",
        },
        "split_manifests": splits,
        "protected_desc_id_manifest_registry": {
            "unique_manifest_count": 2,
            "current_attempt_content_open_operation_count": 0,
            "current_attempt_content_hash_read_count": 0,
            "authority": "AUTH_PROTECTED_ID_MAPPING_ONLY",
            "status": "EXPECTED_HASHES_PRE_REGISTERED_CONTENT_UNOPENED",
            "future_use_requires": "append-only authorized ID-only access intent and receipt",
            "expected_hashes": [
                splits["calib_holdout_desc_id_only"]["expected_sha256"],
                splits["pseudo_official_desc_id_only"]["expected_sha256"],
            ],
        },
        "candidate_corpus_must_remain_full": True,
    }


def collect_protected_registry(
    corpus: dict[str, Any],
    protected_metadata_records: list[dict[str, Any]],
) -> dict[str, Any]:
    desc_paths = {CALIB_HOLDOUT_MANIFEST.resolve(), PSEUDO_OFFICIAL_MANIFEST.resolve()}
    records: list[dict[str, Any]] = []
    for metadata in protected_metadata_records:
        record = dict(metadata)
        resolved = Path(str(record.get("resolved", record.get("declared", ""))))
        if resolved in desc_paths:
            record.update(
                {
                    "access_mode": "STAT_ONLY_WITH_PRE_REGISTERED_EXPECTED_HASH",
                    "content_opened": False,
                    "content_hash_read": False,
                }
            )
        else:
            record.update(
                {
                    "access_mode": "STAT_ONLY_FORBIDDEN_ROOT",
                    "content_opened": False,
                    "content_hash_read": False,
                }
            )
        records.append(record)
    return {
        "schema_version": "c28f_protected_registry_v2",
        "records": records,
        "protected_desc_id_manifest_registry": corpus["protected_desc_id_manifest_registry"],
        "pre_attempt_cumulative_incident": PRE_G0_SECURITY_INCIDENT,
        "pre_attempt_impact_assessment": PRE_G0_SECURITY_INCIDENT_IMPACT_ASSESSMENT,
        "user_acknowledgement": {
            "acknowledged": PRE_G0_SECURITY_INCIDENT_USER_ACK,
            "text_sha256": bytes_sha256(PRE_G0_SECURITY_INCIDENT_USER_ACK_TEXT.encode("utf-8")),
            "disposition": "ACKNOWLEDGED_CONTINUE_WITH_RECORDED_EXCEPTION",
        },
        "current_clean_attempt_access": {
            "protected_sample_content_reads": 0,
            "protected_prediction_content_reads": 0,
            "protected_metric_content_reads": 0,
            "query_text_or_timestamp_reads": 0,
            "protected_model_evaluations": 0,
        },
    }


def _identity_pairs(records: list[dict[str, Any]]) -> set[tuple[int, int]]:
    return {
        (int(record["device"]), int(record["inode"]))
        for record in records
        if record.get("exists") and "device" in record and record.get("type") != "symlink"
    }


def validate_canonical_paths(protected: dict[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Path] = {}
    records: dict[str, Any] = {}
    for name, path in CANONICAL_ROOTS.items():
        if not path.is_absolute() or ".." in path.parts:
            raise PreflightError(f"non-canonical C28F path: {name}={path}")
        chain = _assert_path_chain_no_symlink(path)
        canonical = path.resolve(strict=False)
        resolved[name] = canonical
        final = _stable_metadata_or_missing(path)
        expected_type = "file" if name == "entrypoint" else "directory"
        if final.get("exists") and final.get("type") != expected_type:
            raise PreflightError(f"unexpected canonical root type: {name}: {final.get('type')}")
        records[name] = {"declared": str(path), "resolved": str(canonical), "chain": chain, "final": final}

    containment: list[dict[str, Any]] = []
    names = sorted(resolved)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left, right = resolved[left_name], resolved[right_name]
            overlap = _contains(left, right) or _contains(right, left)
            allowed = {left_name, right_name} == {"reports", "state"} and _contains(
                resolved["reports"], resolved["state"]
            )
            containment.append(
                {"left": left_name, "right": right_name, "overlap": overlap, "allowed": allowed}
            )
            if overlap and not allowed:
                raise PreflightError(f"canonical roots overlap: {left_name}, {right_name}")

    deny_records = [_metadata_or_missing(path) for path in HISTORICAL_DENY_WRITE_ROOTS]
    protected_records = list(protected["records"])
    forbidden_identities = _identity_pairs(deny_records + protected_records)
    for name, target in resolved.items():
        for old_record in deny_records + protected_records:
            old = Path(str(old_record.get("resolved", old_record.get("declared", ""))))
            if old and (_contains(old, target) or _contains(target, old)):
                raise PreflightError(f"canonical root overlaps protected/historical path: {name}: {old}")
        final = records[name]["final"]
        if final.get("exists") and (int(final["device"]), int(final["inode"])) in forbidden_identities:
            raise PreflightError(f"canonical root aliases protected/historical inode: {name}")
    checkpoint_resolved = CHECKPOINT.resolve(strict=True)
    for name, target in resolved.items():
        if _contains(target, checkpoint_resolved) or _contains(checkpoint_resolved, target):
            raise PreflightError(f"checkpoint overlaps output root: {name}")
    return {
        "schema_version": "c28f_canonical_path_map_v2",
        "paths": records,
        "containment_matrix": containment,
        "historical_deny_records": deny_records,
        "protected_record_count": len(protected_records),
    }


def _allowed_control_relative(relative: Path, *, object_type: str) -> bool:
    exact = {
        Path("single_writer.lock"),
        Path("single_writer.heartbeat.json"),
        Path("g0_bootstrap_intent.json"),
        Path("g0_bootstrap_intent.root.sha256"),
        Path("execution_scope.json"),
        Path("authority_ledger.json"),
        Path("evidence_lock.json"),
        Path("baseline_registry.json"),
        Path("gate_registry.json"),
        Path("budget.json"),
        Path("rollback_map.json"),
        Path("preexisting_dirty_inventory.json"),
        Path("artifact_manifest.json"),
        Path("artifact_manifest.root.sha256"),
        Path("GOAL_EVENTS.jsonl"),
        Path("GOAL_STATE.json"),
        Path("CONTINUATION.md"),
    }
    if object_type == "file" and relative in exact:
        return True
    if object_type == "directory":
        if relative in {
            Path("transactions"),
            Path("registry_records"),
            Path("state_records"),
            Path("event_records"),
        }:
            return True
        return (
            len(relative.parts) == 2
            and relative.parts[0] == "transactions"
            and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", relative.parts[1]) is not None
        )
    text = relative.as_posix()
    patterns = (
        r"transactions/[A-Za-z0-9_.-]{1,128}/plan\.json",
        r"registry_records/\d{6}_[A-Za-z0-9_]+_[0-9a-f]{64}\.json",
        r"state_records/\d{6}_[0-9a-f]{64}\.json",
        r"event_records/\d{6}_[0-9a-f]{64}\.jsonl",
    )
    return object_type == "file" and any(re.fullmatch(pattern, text) for pattern in patterns)


def _walk_namespace(root: Path) -> list[tuple[Path, str]]:
    if not os.path.lexists(root):
        return []
    root_identity = lstat_identity(root)
    if root_identity["type"] != "directory":
        raise PreflightError(f"namespace root must be a non-symlink directory: {root}")
    found: list[tuple[Path, str]] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink():
                raise PreflightError(f"symlink in C28F namespace: {path}")
            if entry.is_dir(follow_symlinks=False):
                found.append((path, "directory"))
                stack.append(path)
            elif entry.is_file(follow_symlinks=False):
                found.append((path, "file"))
            else:
                raise PreflightError(f"special filesystem object in C28F namespace: {path}")
    return found


def _recovery_temp_details(relative: Path) -> dict[str, str] | None:
    if not relative.parts:
        return None
    name = relative.name
    match = re.fullmatch(
        r"\.(?P<target>.+)\.(?P<kind>tmp|immutable)\."
        r"(?P<lineage>[0-9a-f]{32})\.(?P<pid>\d+)\.(?P<nonce>[0-9a-f]{32})",
        name,
    )
    if match is None:
        return None
    parent = relative.parent
    if parent != Path(".") and not _allowed_control_relative(parent, object_type="directory"):
        return None
    target_relative = parent / match.group("target")
    if not _allowed_control_relative(target_relative, object_type="file"):
        return None
    return {key: value for key, value in match.groupdict().items() if value is not None}


def _is_recoverable_temp(relative: Path) -> bool:
    return _recovery_temp_details(relative) is not None


def _allowed_quarantine_relative(relative: Path, *, object_type: str) -> bool:
    if relative == Path("quarantine"):
        return object_type == "directory"
    for attempt_id in QUARANTINED_G0_ATTEMPTS:
        attempt_root = Path("quarantine") / attempt_id
        control_root = attempt_root / "goal_control"
        if relative in {attempt_root, control_root}:
            return object_type == "directory"
        if relative == attempt_root / "MIGRATION_RECEIPT.json":
            return object_type == "file"
        if control_root in relative.parents:
            return object_type in {"directory", "file"}
    return False


def validate_namespace_ownership(*, allow_partial_control: bool) -> dict[str, Any]:
    unknown: list[str] = []
    recovery_temps: list[str] = []
    if os.path.lexists(REPORT_ROOT):
        report_entries = _walk_namespace(REPORT_ROOT)
        allowed_temp_lineages: set[str] = set()
        heartbeat_path = CONTROL_ROOT / "single_writer.heartbeat.json"
        if os.path.lexists(heartbeat_path):
            heartbeat = load_json(heartbeat_path)
            if (
                not isinstance(heartbeat, dict)
                or heartbeat.get("schema_version") != "c28f_single_writer_lease_v2"
                or heartbeat.get("goal_id") != GOAL_ID
                or re.fullmatch(r"[0-9a-f]{32}", str(heartbeat.get("lease_lineage_root_id", "")))
                is None
            ):
                raise PreflightError("heartbeat cannot authorize recovery-temp lineage")
            allowed_temp_lineages.add(str(heartbeat["lease_lineage_root_id"]))
        heartbeat_temp_candidates: list[tuple[Path, dict[str, str]]] = []
        all_control_temp_paths: set[Path] = set()
        for candidate, candidate_type in report_entries:
            if candidate_type != "file" or CONTROL_ROOT not in candidate.parents:
                continue
            details = _recovery_temp_details(candidate.relative_to(CONTROL_ROOT))
            if details is not None:
                all_control_temp_paths.add(candidate)
            if (
                details is not None
                and details["target"] == "single_writer.heartbeat.json"
                and candidate.parent == CONTROL_ROOT
            ):
                heartbeat_temp_candidates.append((candidate, details))
        non_temp_control_files = {
            candidate.relative_to(CONTROL_ROOT).as_posix()
            for candidate, candidate_type in report_entries
            if candidate_type == "file"
            and CONTROL_ROOT in candidate.parents
            and _recovery_temp_details(candidate.relative_to(CONTROL_ROOT)) is None
        }
        nested_control_directories = {
            candidate
            for candidate, candidate_type in report_entries
            if candidate_type == "directory" and CONTROL_ROOT in candidate.parents
        }
        restricted_pre_intent_heartbeat_recovery = (
            not os.path.lexists(CONTROL_ROOT / "g0_bootstrap_intent.json")
            and len(heartbeat_temp_candidates) == 1
            and all_control_temp_paths == {heartbeat_temp_candidates[0][0]}
            and non_temp_control_files.issubset(
                {"single_writer.lock", "single_writer.heartbeat.json"}
            )
            and not nested_control_directories
            and lstat_identity(heartbeat_temp_candidates[0][0])["nlink"] == 1
        )
        if restricted_pre_intent_heartbeat_recovery:
            lock_bytes, _lock_record = read_regular_bytes(
                CONTROL_ROOT / "single_writer.lock",
                max_bytes=128,
                require_nlink_one=True,
            )
            if lock_bytes != b"C28F_SINGLE_WRITER_LOCK_V2\n":
                raise PreflightError("pre-intent heartbeat temp lacks a valid fixed lock anchor")
            allowed_temp_lineages.add(heartbeat_temp_candidates[0][1]["lineage"])
        for candidate, candidate_type in report_entries:
            if candidate_type != "file" or CONTROL_ROOT not in candidate.parents:
                continue
            candidate_relative = candidate.relative_to(CONTROL_ROOT)
            details = _recovery_temp_details(candidate_relative)
            if details is None or details["target"] != "single_writer.heartbeat.json":
                continue
            try:
                candidate_payload = load_json(candidate)
            except Exception:
                continue
            if (
                isinstance(candidate_payload, dict)
                and candidate_payload.get("schema_version") == "c28f_single_writer_lease_v2"
                and candidate_payload.get("goal_id") == GOAL_ID
                and candidate_payload.get("lease_lineage_root_id") == details["lineage"]
            ):
                allowed_temp_lineages.add(details["lineage"])
        valid_temp_paths = {
            candidate
            for candidate, candidate_type in report_entries
            if candidate_type == "file"
            and CONTROL_ROOT in candidate.parents
            and (details := _recovery_temp_details(candidate.relative_to(CONTROL_ROOT))) is not None
            and details["lineage"] in allowed_temp_lineages
        }
        control_files_by_inode: dict[tuple[int, int], list[Path]] = {}
        for candidate, candidate_type in report_entries:
            if candidate_type == "file" and CONTROL_ROOT in candidate.parents:
                identity = lstat_identity(candidate)
                control_files_by_inode.setdefault(
                    (int(identity["device"]), int(identity["inode"])), []
                ).append(candidate)
        for path, object_type in report_entries:
            relative = path.relative_to(REPORT_ROOT)
            if relative == Path("bootstrap.lock"):
                if object_type != "file":
                    unknown.append(relative.as_posix())
                else:
                    lock_bytes, lock_record = read_regular_bytes(
                        path, max_bytes=256, require_nlink_one=True
                    )
                    if lock_bytes != BOOTSTRAP_LOCK_HEADER or lock_record["nlink"] != 1:
                        unknown.append(relative.as_posix())
            elif relative.parts[0] == "quarantine":
                if not _allowed_quarantine_relative(relative, object_type=object_type):
                    unknown.append(relative.as_posix())
            elif relative.parts[0] != "goal_control":
                unknown.append(relative.as_posix())
            elif relative == Path("goal_control"):
                if object_type != "directory":
                    unknown.append(relative.as_posix())
            else:
                control_relative = path.relative_to(CONTROL_ROOT)
                if object_type == "file":
                    identity = lstat_identity(path)
                    nlink = int(identity["nlink"])
                    if nlink != 1:
                        peers = control_files_by_inode.get(
                            (int(identity["device"]), int(identity["inode"])), []
                        )
                        temp_peers = [
                            peer
                            for peer in peers
                            if peer in valid_temp_paths
                        ]
                        non_temp_peers = [peer for peer in peers if peer not in temp_peers]
                        valid_immutable_install_window = (
                            nlink == 2
                            and len(peers) == 2
                            and len(temp_peers) == 1
                            and len(non_temp_peers) == 1
                            and temp_peers[0].parent == non_temp_peers[0].parent
                            and non_temp_peers[0] != CONTROL_ROOT / "single_writer.lock"
                        )
                        if not valid_immutable_install_window:
                            raise PreflightError(f"unrecognized hardlink in C28F namespace: {path}")
                        recovery_temps.append(temp_peers[0].relative_to(CONTROL_ROOT).as_posix())
                if object_type == "file" and _is_recoverable_temp(control_relative):
                    if path in valid_temp_paths:
                        recovery_temps.append(control_relative.as_posix())
                    else:
                        unknown.append(relative.as_posix())
                elif not _allowed_control_relative(control_relative, object_type=object_type):
                    unknown.append(relative.as_posix())
    for root in (CACHE_ROOT, CHECKPOINT_ROOT):
        if os.path.lexists(root) and _walk_namespace(root):
            unknown.append(str(root.relative_to(REPO_ROOT)) + "/<nonempty>")
    if unknown:
        raise PreflightError(f"unknown pre-existing C28F namespace content: {sorted(unknown)[:20]}")
    control_files = []
    control_directories: list[str] = []
    if os.path.lexists(CONTROL_ROOT):
        control_entries = _walk_namespace(CONTROL_ROOT)
        control_files = sorted(
            path.relative_to(CONTROL_ROOT).as_posix()
            for path, object_type in control_entries
            if object_type == "file" and not _is_recoverable_temp(path.relative_to(CONTROL_ROOT))
        )
        control_directories = sorted(
            path.relative_to(CONTROL_ROOT).as_posix()
            for path, object_type in control_entries
            if object_type == "directory"
        )
        if not allow_partial_control and any(
            name not in {"single_writer.lock", "single_writer.heartbeat.json"} for name in control_files
        ):
            raise PreflightError("unexpected partial G0 control state before new intent")
        if not allow_partial_control and (control_directories or recovery_temps):
            raise PreflightError("unexpected directory/temp residue before new G0 intent")
    return {
        "unknown": [],
        "control_files": control_files,
        "control_directories": control_directories,
        "recovery_temps": sorted(set(recovery_temps)),
        "recovery_temp_identities": {
            relative: _stable_identity_fields(lstat_identity(CONTROL_ROOT / relative))
            for relative in sorted(set(recovery_temps))
        },
    }


def _migration_identity(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in ("type", "device", "inode", "mode", "nlink")
        if key in record
    }


def _quarantined_g0_tree_record(
    root: Path,
    attempt_id: str,
    expected: dict[str, Any],
    protected_records: list[dict[str, Any]],
) -> dict[str, Any]:
    forbidden = _identity_pairs(protected_records)
    entries = _walk_namespace(root)
    directories = sorted(
        path.relative_to(root).as_posix()
        for path, object_type in entries
        if object_type == "directory"
    )
    files = sorted(
        path.relative_to(root).as_posix()
        for path, object_type in entries
        if object_type == "file"
    )
    if len(files) != expected["file_count"] or len(directories) != expected["directory_count"]:
        raise PreflightError(f"quarantined G0 tree cardinality mismatch: {attempt_id}")
    tree_lines = [f"D\t{relative}\n".encode("utf-8") for relative in directories]
    file_records: list[dict[str, Any]] = []
    for relative in files:
        path = root / relative
        identity = lstat_identity(path)
        if identity["type"] != "file" or int(identity["nlink"]) != 1:
            raise PreflightError(f"quarantined G0 file is not single-link regular: {path}")
        if (int(identity["device"]), int(identity["inode"])) in forbidden:
            raise PreflightError(f"quarantined G0 file aliases protected content: {path}")
        _data, record = read_regular_bytes(path, require_nlink_one=True)
        tree_lines.append(
            f"F\t{relative}\t{record['size_bytes']}\t{record['sha256']}\n".encode("utf-8")
        )
        file_records.append(
            {
                "path": relative,
                "size_bytes": record["size_bytes"],
                "sha256": record["sha256"],
            }
        )
    tree_sha256 = bytes_sha256(b"".join(tree_lines))
    if tree_sha256 != expected["control_tree_sha256"]:
        raise PreflightError(f"quarantined G0 tree hash mismatch: {attempt_id}")
    file_by_path = {item["path"]: item for item in file_records}
    if (
        file_by_path.get("g0_bootstrap_intent.json", {}).get("sha256")
        != expected["intent_file_sha256"]
        or file_by_path.get("budget.json", {}).get("sha256")
        != expected["observed_budget_file_sha256"]
    ):
        raise PreflightError(f"quarantined G0 key-file hash mismatch: {attempt_id}")

    intent = load_json(root / "g0_bootstrap_intent.json")
    plan = load_json(root / "transactions" / expected["transaction_id"] / "plan.json")
    state = load_json(root / "GOAL_STATE.json")
    budget = load_json(root / "budget.json")
    evidence = load_json(root / "evidence_lock.json")
    journal, _journal_record = read_regular_bytes(
        root / "GOAL_EVENTS.jsonl", require_nlink_one=True
    )
    if not journal.endswith(b"\n") or journal.count(b"\n") != 1:
        raise PreflightError(f"quarantined G0 event framing mismatch: {attempt_id}")
    event = json.loads(journal[:-1].decode("utf-8"))
    if canonical_json_bytes(event) != journal[:-1] or event.get("event_sha256") != semantic_sha256(
        event, excluded_fields=("event_sha256",)
    ):
        raise PreflightError(f"quarantined G0 event canonical hash mismatch: {attempt_id}")
    receipt_sha = (
        intent.get("immutable_lock", {})
        .get("static_review_receipt", {})
        .get("file", {})
        .get("sha256")
    )
    if (
        intent.get("intent_sha256")
        != semantic_sha256(intent, excluded_fields=("intent_sha256",))
        or plan.get("plan_sha256")
        != semantic_sha256(plan, excluded_fields=("plan_sha256",))
        or state.get("state_sha256")
        != semantic_sha256(state, excluded_fields=("state_sha256",))
        or intent.get("goal_id") != GOAL_ID
        or intent.get("attempt_id") != attempt_id
        or intent.get("transaction_id") != expected["transaction_id"]
        or plan.get("goal_id") != GOAL_ID
        or plan.get("attempt_id") != attempt_id
        or plan.get("transaction_id") != expected["transaction_id"]
        or state.get("goal_id") != GOAL_ID
        or state.get("attempt_id") != attempt_id
        or state.get("transaction_id") != expected["transaction_id"]
        or event.get("attempt_id") != attempt_id
        or event.get("transaction_id") != expected["transaction_id"]
        or state.get("state_sha256") != expected["observed_state_sha256"]
        or plan.get("expected_state_sha256") != expected["observed_state_sha256"]
        or event.get("expected_next_state_sha256") != expected["observed_state_sha256"]
        or event.get("event_sha256") != expected["observed_event_sha256"]
        or event.get("output_hashes", {}).get("budget.json")
        != expected["observed_budget_file_sha256"]
        or receipt_sha != expected["static_review_receipt_sha256"]
    ):
        raise PreflightError(f"quarantined G0 attempt/transaction lineage mismatch: {attempt_id}")
    usage = budget.get("usage", {})
    tokens = budget.get("eval_tokens", {})
    clean_access = evidence.get("current_clean_attempt_access", {})
    if (
        state.get("status") != "PREFLIGHT_OK"
        or state.get("real_data_optimizer_updates") != 0
        or state.get("protected_model_evals") != 0
        or state.get("eval_token_states") != {"F0_A": "UNISSUED", "F0_B": "UNISSUED"}
        or not isinstance(usage, dict)
        or any(type(value) is not int or value != 0 for value in usage.values())
        or set(tokens) != {"F0_A", "F0_B"}
        or any(token.get("state") != "UNISSUED" for token in tokens.values())
        or not isinstance(clean_access, dict)
        or any(type(value) is not int or value != 0 for value in clean_access.values())
    ):
        raise PreflightError(f"quarantined G0 zero-evaluation/zero-update invariant mismatch: {attempt_id}")
    root_identity = lstat_identity(root)
    old_lock_identity = lstat_identity(root / "single_writer.lock")
    return {
        **expected,
        "status": "QUARANTINED_INVALID_CONTROL_TRANSACTION",
        "nominal_old_state_invalidated": "PREFLIGHT_OK_REJECTED_BY_FINAL_SELF_VERIFIER",
        "path": str(root),
        "tree_hash_domain": "sorted directory lines, then sorted path/size/full-sha file lines",
        "observed_control_tree_sha256": tree_sha256,
        "root_identity": _migration_identity(root_identity),
        "old_writer_lock_identity": _migration_identity(old_lock_identity),
        "files": file_records,
    }


def collect_quarantined_g0_attempts(
    protected_records: list[dict[str, Any]],
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for attempt_id in sorted(QUARANTINED_G0_ATTEMPTS):
        expected = QUARANTINED_G0_ATTEMPTS[attempt_id]
        attempt_root = QUARANTINE_ROOT / attempt_id
        tree = _quarantined_g0_tree_record(
            attempt_root / "goal_control", attempt_id, expected, protected_records
        )
        receipt_path = attempt_root / "MIGRATION_RECEIPT.json"
        _assert_content_source_not_protected(receipt_path, protected_records)
        receipt = load_json(receipt_path)
        if not isinstance(receipt, dict) or set(receipt) != {
            "schema_version",
            "status",
            "goal_id",
            "attempt_id",
            "transaction_id",
            "source_control_root",
            "quarantine_control_root",
            "control_tree_sha256",
            "control_tree_file_count",
            "control_tree_directory_count",
            "preserved_root_identity",
            "old_writer_lock_identity",
            "fresh_writer_lock_identity",
            "bootstrap_lock_identity",
            "migration_command_argv_sha256",
            "migration_static_review_receipt_sha256",
            "migration_method",
            "migration_observation",
            "target_precondition",
            "same_filesystem",
            "source_parent_fsynced",
            "target_parent_fsynced",
            "root_cause",
            "failure_fingerprint",
            "failure_stage",
            "scientific_impact",
            "model_forward_evaluations",
            "protected_content_opens_current_attempt",
            "nominal_old_state_invalidated",
            "receipt_sha256",
        }:
            raise PreflightError(f"quarantine migration receipt schema mismatch: {attempt_id}")
        if receipt.get("receipt_sha256") != semantic_sha256(
            receipt, excluded_fields=("receipt_sha256",)
        ):
            raise PreflightError(f"quarantine migration receipt self-hash mismatch: {attempt_id}")
        fresh_lock_identity = _migration_identity(lstat_identity(CONTROL_ROOT / "single_writer.lock"))
        bootstrap_lock_identity = _migration_identity(lstat_identity(BOOTSTRAP_LOCK))
        if (
            receipt.get("schema_version") != "c28f_g0_quarantine_migration_v1"
            or receipt.get("status") != "QUARANTINED_INVALID_CONTROL_TRANSACTION"
            or receipt.get("goal_id") != GOAL_ID
            or receipt.get("attempt_id") != attempt_id
            or receipt.get("transaction_id") != expected["transaction_id"]
            or receipt.get("control_tree_sha256") != expected["control_tree_sha256"]
            or receipt.get("control_tree_file_count") != expected["file_count"]
            or receipt.get("control_tree_directory_count") != expected["directory_count"]
            or receipt.get("preserved_root_identity") != tree["root_identity"]
            or receipt.get("old_writer_lock_identity") != tree["old_writer_lock_identity"]
            or receipt.get("fresh_writer_lock_identity") != fresh_lock_identity
            or receipt.get("bootstrap_lock_identity") != bootstrap_lock_identity
            or not isinstance(receipt.get("migration_command_argv_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", receipt["migration_command_argv_sha256"]) is None
            or receipt.get("migration_static_review_receipt_sha256")
            != regular_file_record(STATIC_REVIEW_RECEIPT, content_hash=True)["sha256"]
            or receipt.get("source_control_root") != str(CONTROL_ROOT)
            or receipt.get("quarantine_control_root") != str(attempt_root / "goal_control")
            or receipt.get("root_cause") != expected["root_cause"]
            or receipt.get("failure_fingerprint") != expected["failure_fingerprint"]
            or receipt.get("failure_stage") != expected["failure_stage"]
            or receipt.get("scientific_impact") != "NONE"
            or receipt.get("model_forward_evaluations") != 0
            or receipt.get("protected_content_opens_current_attempt") != 0
            or receipt.get("migration_method") != "renameat2(RENAME_NOREPLACE)"
            or receipt.get("migration_observation")
            not in {"ATOMIC_RENAME_EXECUTED_THIS_ACTION", "EXACT_POSTIMAGE_RECOVERED"}
            or receipt.get("target_precondition") != "MUST_BE_ABSENT"
            or receipt.get("same_filesystem") is not True
            or receipt.get("source_parent_fsynced") is not True
            or receipt.get("target_parent_fsynced") is not True
            or receipt.get("nominal_old_state_invalidated")
            != "PREFLIGHT_OK_REJECTED_BY_FINAL_SELF_VERIFIER"
        ):
            raise PreflightError(f"quarantine migration receipt lineage mismatch: {attempt_id}")
        receipt_record = regular_file_record(receipt_path, content_hash=True)
        records[attempt_id] = {**tree, "migration_receipt": receipt, "migration_receipt_file": receipt_record}
    return records


def _assert_quarantine_business_closure(
    business: dict[str, Any], quarantined: dict[str, Any]
) -> None:
    prefix = QUARANTINE_ROOT.relative_to(REPO_ROOT).as_posix() + "/"
    actual = {
        item["path"]: {"size_bytes": item["size_bytes"], "sha256": item["sha256"]}
        for item in business["entries"]
        if item["path"].startswith(prefix)
    }
    expected: dict[str, dict[str, Any]] = {}
    for attempt_id in sorted(quarantined):
        record = quarantined[attempt_id]
        root = Path(record["path"])
        for item in record["files"]:
            path = (root / item["path"]).relative_to(REPO_ROOT).as_posix()
            expected[path] = {"size_bytes": item["size_bytes"], "sha256": item["sha256"]}
        receipt = record["migration_receipt_file"]
        expected[Path(receipt["path"]).relative_to(REPO_ROOT).as_posix()] = {
            "size_bytes": receipt["size_bytes"],
            "sha256": receipt["sha256"],
        }
    if actual != expected:
        raise PreflightError("quarantine/business-inventory exact file closure mismatch")


def _renameat2_noreplace(
    source_parent_fd: int,
    source_name: bytes,
    target_parent_fd: int,
    target_name: bytes,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise PreflightError("renameat2 is unavailable; quarantine migration cannot be atomic no-replace")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(source_parent_fd, source_name, target_parent_fd, target_name, 1) != 0:
        failure_errno = ctypes.get_errno()
        raise OSError(failure_errno, os.strerror(failure_errno))


def _open_directory_fd(path: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags)


def _open_directory_at(parent_fd: int, name: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(name, flags, dir_fd=parent_fd)


def _ensure_directory_at(parent_fd: int, name: str) -> int:
    try:
        os.mkdir(name, mode=0o750, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except FileExistsError:
        pass
    return _open_directory_at(parent_fd, name)


def _write_all_fd(fd: int, data: bytes) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        written = os.write(fd, view[offset:])
        if written <= 0:
            raise OSError("migration receipt write made no progress")
        offset += written


def _read_fd_bytes(fd: int, max_bytes: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(fd, min(1024 * 1024, max_bytes - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise PreflightError("migration file exceeds its fixed read bound")
        chunks.append(chunk)
    return b"".join(chunks)


def _heartbeat_owner_is_same_live_process(heartbeat: dict[str, Any]) -> bool:
    current_boot = _proc_read(Path("/proc/sys/kernel/random/boot_id"), 256).decode(
        "ascii", "strict"
    ).strip()
    if heartbeat.get("boot_id") != current_boot:
        return False
    pid = heartbeat.get("pid")
    start_ticks = heartbeat.get("process_start_ticks")
    if type(pid) is not int or type(start_ticks) is not int:
        raise PreflightError("quarantined heartbeat lacks PID/start-ticks identity")
    try:
        stat_bytes = _proc_read(Path(f"/proc/{pid}/stat"), 8192)
    except FileNotFoundError:
        return False
    fields = stat_bytes.decode("utf-8", "strict").split()
    if len(fields) <= 21:
        raise PreflightError("live PID stat is too short for quarantine decision")
    return int(fields[21]) == start_ticks


def _install_migration_receipt(attempt_fd: int, receipt: dict[str, Any]) -> None:
    data = canonical_json_bytes(receipt) + b"\n"
    target = "MIGRATION_RECEIPT.json"
    temp = ".MIGRATION_RECEIPT.json.tmp"
    flags_read = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags_read |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags_read |= os.O_NOFOLLOW
    try:
        target_fd = os.open(target, flags_read, dir_fd=attempt_fd)
    except FileNotFoundError:
        target_fd = None
    if target_fd is not None:
        try:
            target_stat = os.fstat(target_fd)
            if not stat.S_ISREG(target_stat.st_mode) or target_stat.st_nlink != 1:
                raise PreflightError("migration receipt target is not single-link regular")
            if _read_fd_bytes(target_fd, 1024 * 1024) != data:
                raise PreflightError("existing migration receipt differs from the deterministic receipt")
        finally:
            os.close(target_fd)
        try:
            os.unlink(temp, dir_fd=attempt_fd)
            os.fsync(attempt_fd)
        except FileNotFoundError:
            pass
        return
    try:
        stale_fd = os.open(temp, flags_read, dir_fd=attempt_fd)
    except FileNotFoundError:
        stale_fd = None
    if stale_fd is not None:
        try:
            stale_stat = os.fstat(stale_fd)
            if not stat.S_ISREG(stale_stat.st_mode) or stale_stat.st_nlink != 1:
                raise PreflightError("migration receipt temp is not single-link regular")
            stale_data = _read_fd_bytes(stale_fd, 1024 * 1024)
        finally:
            os.close(stale_fd)
        if stale_data != data:
            os.unlink(temp, dir_fd=attempt_fd)
            os.fsync(attempt_fd)
    flags_write = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags_write |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags_write |= os.O_NOFOLLOW
    try:
        temp_fd = os.open(temp, flags_write, 0o600, dir_fd=attempt_fd)
    except FileExistsError:
        temp_fd = None
    if temp_fd is not None:
        try:
            _write_all_fd(temp_fd, data)
            os.fsync(temp_fd)
        finally:
            os.close(temp_fd)
    _renameat2_noreplace(attempt_fd, temp.encode("ascii"), attempt_fd, target.encode("ascii"))
    os.fsync(attempt_fd)


def _build_migration_receipt(
    *,
    attempt_id: str,
    expected: dict[str, Any],
    tree: dict[str, Any],
    fresh_lock_identity: dict[str, Any],
    bootstrap_lock_identity: dict[str, Any],
    producer_argv: list[str],
    static_review_receipt_sha256: str,
    migration_observation: str,
) -> dict[str, Any]:
    base = {
        "schema_version": "c28f_g0_quarantine_migration_v1",
        "status": "QUARANTINED_INVALID_CONTROL_TRANSACTION",
        "goal_id": GOAL_ID,
        "attempt_id": attempt_id,
        "transaction_id": expected["transaction_id"],
        "source_control_root": str(CONTROL_ROOT),
        "quarantine_control_root": str(QUARANTINE_ROOT / attempt_id / "goal_control"),
        "control_tree_sha256": expected["control_tree_sha256"],
        "control_tree_file_count": expected["file_count"],
        "control_tree_directory_count": expected["directory_count"],
        "preserved_root_identity": tree["root_identity"],
        "old_writer_lock_identity": tree["old_writer_lock_identity"],
        "fresh_writer_lock_identity": fresh_lock_identity,
        "bootstrap_lock_identity": bootstrap_lock_identity,
        "migration_command_argv_sha256": bytes_sha256(canonical_json_bytes(producer_argv)),
        "migration_static_review_receipt_sha256": static_review_receipt_sha256,
        "migration_method": "renameat2(RENAME_NOREPLACE)",
        "migration_observation": migration_observation,
        "target_precondition": "MUST_BE_ABSENT",
        "same_filesystem": True,
        "source_parent_fsynced": True,
        "target_parent_fsynced": True,
        "root_cause": expected["root_cause"],
        "failure_fingerprint": expected["failure_fingerprint"],
        "failure_stage": expected["failure_stage"],
        "scientific_impact": "NONE",
        "model_forward_evaluations": 0,
        "protected_content_opens_current_attempt": 0,
        "nominal_old_state_invalidated": "PREFLIGHT_OK_REJECTED_BY_FINAL_SELF_VERIFIER",
    }
    return {**base, "receipt_sha256": bytes_sha256(canonical_json_bytes(base))}


def quarantine_failed_g0_attempt(
    producer_argv: list[str],
    producer_raw_argv: list[str],
    bootstrap_lease: BootstrapLease,
) -> dict[str, Any]:
    bootstrap_lease.assert_active_capability()
    validate_bootstrap_control_parent()
    if (
        len(producer_argv) != 5
        or producer_argv[1] != "-B"
        or producer_argv[-2:] != ["--action", "G0_QUARANTINE_FAILED_BOOTSTRAP"]
        or len(producer_raw_argv) != 5
        or producer_raw_argv[1] != "-B"
        or producer_raw_argv[-2:] != ["--action", "G0_QUARANTINE_FAILED_BOOTSTRAP"]
        or Path(producer_raw_argv[0]).resolve(strict=True) != Path(producer_argv[0])
        or Path(producer_raw_argv[2]).resolve(strict=True) != ENTRYPOINT
    ):
        raise PreflightError("quarantine action lacks a canonical real -B invocation")
    if len(QUARANTINED_G0_ATTEMPTS) != 1:
        raise PreflightError("quarantine migration requires exactly one preregistered failed attempt")
    attempt_id = next(iter(QUARANTINED_G0_ATTEMPTS))
    expected = QUARANTINED_G0_ATTEMPTS[attempt_id]
    destination_root = QUARANTINE_ROOT / attempt_id / "goal_control"
    source_has_old_intent = os.path.lexists(CONTROL_ROOT / "g0_bootstrap_intent.json")
    destination_exists = os.path.lexists(destination_root)
    renamed_now = False
    if source_has_old_intent and destination_exists:
        raise PreflightError("both live old G0 intent and quarantine destination exist")
    if not source_has_old_intent and not destination_exists:
        raise PreflightError("neither live old G0 intent nor quarantine destination exists")

    protected_records = collect_protected_metadata_records()
    _assert_control_namespace_not_protected(protected_records)
    static_receipt = collect_static_review_receipt(protected_records)
    lock_root = CONTROL_ROOT if source_has_old_intent else destination_root
    old_lock_path = lock_root / "single_writer.lock"
    flags = os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    old_lock_fd = os.open(old_lock_path, flags)
    fresh_lock_fd: int | None = None
    report_fd: int | None = None
    quarantine_fd: int | None = None
    attempt_fd: int | None = None
    try:
        fcntl.flock(old_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old_lock_stat = os.fstat(old_lock_fd)
        if (
            not stat.S_ISREG(old_lock_stat.st_mode)
            or old_lock_stat.st_nlink != 1
            or _read_fd_bytes(old_lock_fd, 256) != b"C28F_SINGLE_WRITER_LOCK_V2\n"
        ):
            raise PreflightError("old G0 writer lock is not the fixed single-link lock")
        heartbeat = load_json(lock_root / "single_writer.heartbeat.json")
        if heartbeat.get("status") != "INTERRUPTED" or _heartbeat_owner_is_same_live_process(heartbeat):
            raise PreflightError("old G0 writer is not proven interrupted and dead")
        pre_tree = _quarantined_g0_tree_record(
            lock_root, attempt_id, expected, protected_records
        )
        if (
            pre_tree["old_writer_lock_identity"] != _migration_identity(lstat_identity(old_lock_path))
            or (int(old_lock_stat.st_dev), int(old_lock_stat.st_ino))
            != (
                int(pre_tree["old_writer_lock_identity"]["device"]),
                int(pre_tree["old_writer_lock_identity"]["inode"]),
            )
        ):
            raise PreflightError("old G0 writer lock identity drift before migration")

        report_fd = _open_directory_fd(REPORT_ROOT)
        quarantine_fd = _ensure_directory_at(report_fd, "quarantine")
        attempt_fd = _ensure_directory_at(quarantine_fd, attempt_id)
        if (
            os.fstat(report_fd).st_dev != os.fstat(attempt_fd).st_dev
            or lstat_identity(lock_root)["device"] != os.fstat(attempt_fd).st_dev
        ):
            raise PreflightError("quarantine migration crosses filesystems")
        if source_has_old_intent:
            try:
                os.stat("goal_control", dir_fd=attempt_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise PreflightError("quarantine target appeared before no-replace rename")
            _renameat2_noreplace(report_fd, b"goal_control", attempt_fd, b"goal_control")
            renamed_now = True
            os.fsync(report_fd)
            os.fsync(attempt_fd)
            os.fsync(quarantine_fd)
        try:
            os.stat(CONTROL_ROOT, follow_symlinks=False)
        except FileNotFoundError:
            os.mkdir("goal_control", mode=0o750, dir_fd=report_fd)
            os.fsync(report_fd)
        else:
            if os.path.lexists(CONTROL_ROOT / "g0_bootstrap_intent.json"):
                raise PreflightError("fresh canonical control root still contains the old intent")
            partial_entries = _walk_namespace(CONTROL_ROOT)
            allowed_partial = {
                (CONTROL_ROOT / "single_writer.lock", "file"),
            }
            if set(partial_entries) - allowed_partial:
                raise PreflightError("fresh canonical control root contains unrecognized migration residue")
        fresh_root_fd = _open_directory_fd(CONTROL_ROOT)
        try:
            fresh_flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_CLOEXEC"):
                fresh_flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                fresh_flags |= os.O_NOFOLLOW
            fresh_lock_fd = os.open("single_writer.lock", fresh_flags, 0o600, dir_fd=fresh_root_fd)
            fcntl.flock(fresh_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fresh_stat = os.fstat(fresh_lock_fd)
            if not stat.S_ISREG(fresh_stat.st_mode) or fresh_stat.st_nlink != 1:
                raise PreflightError("fresh writer lock is not single-link regular")
            fresh_data = _read_fd_bytes(fresh_lock_fd, 256)
            fresh_header = b"C28F_SINGLE_WRITER_LOCK_V2\n"
            if not fresh_header.startswith(fresh_data):
                raise PreflightError("fresh writer lock header mismatch")
            if fresh_data != fresh_header:
                os.lseek(fresh_lock_fd, len(fresh_data), os.SEEK_SET)
                _write_all_fd(fresh_lock_fd, fresh_header[len(fresh_data) :])
                os.fsync(fresh_lock_fd)
                fresh_data = _read_fd_bytes(fresh_lock_fd, 256)
                fresh_stat = os.fstat(fresh_lock_fd)
            if fresh_data != fresh_header:
                raise PreflightError("fresh writer lock header mismatch")
            fresh_path_stat = os.stat(
                "single_writer.lock", dir_fd=fresh_root_fd, follow_symlinks=False
            )
            if (fresh_stat.st_dev, fresh_stat.st_ino) != (
                fresh_path_stat.st_dev,
                fresh_path_stat.st_ino,
            ):
                raise PreflightError("fresh writer lock path/inode mismatch")
            os.fsync(fresh_root_fd)
        finally:
            os.close(fresh_root_fd)

        os.fsync(report_fd)
        os.fsync(attempt_fd)
        os.fsync(quarantine_fd)
        post_tree = _quarantined_g0_tree_record(
            destination_root, attempt_id, expected, protected_records
        )
        if (
            post_tree["root_identity"] != pre_tree["root_identity"]
            or post_tree["old_writer_lock_identity"] != pre_tree["old_writer_lock_identity"]
        ):
            raise PreflightError("quarantine rename did not preserve root/old-lock inode identity")
        if os.path.lexists(QUARANTINE_ROOT / attempt_id / "MIGRATION_RECEIPT.json"):
            verified = collect_quarantined_g0_attempts(protected_records)
            bootstrap_lease.assert_active_capability()
            return {
                "status": "ALREADY_QUARANTINED_EXACT_POSTIMAGE_VERIFIED",
                "attempt_id": attempt_id,
                "transaction_id": expected["transaction_id"],
                "control_tree_sha256": verified[attempt_id]["observed_control_tree_sha256"],
                "migration_receipt_sha256": verified[attempt_id]["migration_receipt"][
                    "receipt_sha256"
                ],
                "scientific_impact": "NONE",
                "next_action": "G0_INIT",
            }
        bootstrap_identity = _migration_identity(lstat_identity(BOOTSTRAP_LOCK))
        fresh_identity = _migration_identity(lstat_identity(CONTROL_ROOT / "single_writer.lock"))
        receipt = _build_migration_receipt(
            attempt_id=attempt_id,
            expected=expected,
            tree=post_tree,
            fresh_lock_identity=fresh_identity,
            bootstrap_lock_identity=bootstrap_identity,
            producer_argv=producer_argv,
            static_review_receipt_sha256=static_receipt["file"]["sha256"],
            migration_observation=(
                "ATOMIC_RENAME_EXECUTED_THIS_ACTION"
                if renamed_now
                else "EXACT_POSTIMAGE_RECOVERED"
            ),
        )
        _install_migration_receipt(attempt_fd, receipt)
        os.fsync(report_fd)
        os.fsync(attempt_fd)
        os.fsync(quarantine_fd)
        verified = collect_quarantined_g0_attempts(protected_records)
        if static_receipt["file"]["sha256"] != regular_file_record(
            STATIC_REVIEW_RECEIPT, content_hash=True
        )["sha256"]:
            raise PreflightError("static review receipt drifted during quarantine migration")
        bootstrap_lease.assert_active_capability()
        return {
            "status": "QUARANTINED_INVALID_CONTROL_TRANSACTION",
            "attempt_id": attempt_id,
            "transaction_id": expected["transaction_id"],
            "control_tree_sha256": verified[attempt_id]["observed_control_tree_sha256"],
            "migration_receipt_sha256": verified[attempt_id]["migration_receipt"]["receipt_sha256"],
            "scientific_impact": "NONE",
            "next_action": "G0_INIT",
        }
    finally:
        if fresh_lock_fd is not None:
            try:
                fcntl.flock(fresh_lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(fresh_lock_fd)
        for fd in (attempt_fd, quarantine_fd, report_fd):
            if fd is not None:
                os.close(fd)
        try:
            fcntl.flock(old_lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(old_lock_fd)


def cleanup_recovery_temps(*, lease: WriterLease) -> list[str]:
    lease.assert_active_capability()
    snapshot = validate_namespace_ownership(allow_partial_control=True)
    removed: list[str] = []
    for relative_text in snapshot["recovery_temps"]:
        relative = Path(relative_text)
        if not _is_recoverable_temp(relative):
            raise PreflightError(f"refusing to clean unrecognized residue: {relative}")
        path = CONTROL_ROOT / relative
        assert_allowed_write_path(path)
        identity = lstat_identity(path)
        if identity["type"] != "file" or identity["nlink"] not in {1, 2}:
            raise PreflightError(f"recovery temp has an invalid file identity: {path}")
        if _stable_identity_fields(identity) != snapshot["recovery_temp_identities"][relative_text]:
            raise PreflightError(f"recovery temp identity changed before cleanup: {path}")
        path.unlink()
        fsync_directory(path.parent)
        removed.append(relative.as_posix())
    return removed


def lock_checkpoint(protected_records: list[dict[str, Any]]) -> dict[str, Any]:
    _assert_content_source_not_protected(CHECKPOINT, protected_records)
    record = regular_file_record(CHECKPOINT, content_hash=True)
    if record["sha256"] != EXPECTED_CHECKPOINT_SHA256 or record["size_bytes"] != EXPECTED_CHECKPOINT_SIZE:
        raise PreflightError("frozen checkpoint hash/size mismatch")
    return {
        "read_mode": "O_RDONLY|O_NOFOLLOW",
        "metadata_and_hash": record,
        "stable_during_hash": True,
        "mode_was_not_changed": True,
    }


def collect_authority_files(protected_records: list[dict[str, Any]]) -> dict[str, Any]:
    for path in (GOAL_CONTRACT, BLUEPRINT, HANDOFF):
        _assert_content_source_not_protected(path, protected_records)
    records = {
        "goal_contract": regular_file_record(GOAL_CONTRACT, content_hash=True),
        "blueprint": regular_file_record(BLUEPRINT, content_hash=True),
        "handoff": regular_file_record(HANDOFF, content_hash=True),
    }
    expected = {
        "goal_contract": EXPECTED_GOAL_CONTRACT_SHA256,
        "blueprint": EXPECTED_BLUEPRINT_SHA256,
        "handoff": EXPECTED_HANDOFF_SHA256,
    }
    for name, digest in expected.items():
        if records[name]["sha256"] != digest:
            raise PreflightError(f"authority hash mismatch: {name}")
    return records


def collect_legacy_source_hashes(protected_records: list[dict[str, Any]]) -> dict[str, Any]:
    sources = {
        "split_manager_source": REPO_ROOT / "blueprint_e2e_v2/data/split_manager.py",
        "legacy_evaluator_source": REPO_ROOT / "blueprint_e2e_v2/engine/evaluate.py",
        "legacy_metric_nms_source": REPO_ROOT / "blueprint_e2e_v2/engine/metrics.py",
        "legacy_checkpoint_source": REPO_ROOT / "blueprint_e2e_v2/engine/checkpoint.py",
    }
    for path in sources.values():
        _assert_content_source_not_protected(path, protected_records)
    return {name: regular_file_record(path, content_hash=True) for name, path in sources.items()}


def collect_historical_artifacts(protected: dict[str, Any]) -> dict[str, Any]:
    safe_paths = dict(HISTORICAL_ARTIFACTS)
    safe_paths.update(
        {
            "c7_train_calib_predictions": REPO_ROOT
            / "results/rlem_c7_b3/freeze/calibrated_A4_predictions.jsonl",
            "c7_train_calib_scores": REPO_ROOT / "results/rlem_c7_b3/freeze/calibrated_A4_scores.jsonl",
            "c7_train_calib_cache": REPO_ROOT / "results/rlem_c6a/cache/train_calib_c6_cache.npz",
        }
    )
    protected_ids = _identity_pairs(list(protected["records"]))
    records: dict[str, Any] = {}
    for name, path in safe_paths.items():
        _assert_content_source_not_protected(path, list(protected["records"]))
        metadata = _metadata_or_missing(path)
        if not metadata.get("exists") or metadata.get("type") != "file":
            raise PreflightError(f"safe historical artifact missing: {name}")
        identity = (int(metadata["device"]), int(metadata["inode"]))
        if identity in protected_ids:
            raise PreflightError(f"historical replay artifact aliases a protected inode: {name}")
        records[name] = regular_file_record(path, content_hash=True)
    expected = {
        "c7_train_calib_predictions": "7e20f32de6b54c8a98f17eea3dd0e2d84db08896047d0e85f96a3fafaeb71e16",
        "c7_train_calib_scores": "9d0c1964581a78a81480d9c648f67f0de60334f5c66fa8b35465c6ac43b810ab",
        "c7_train_calib_cache": "4aa5f0e5943bd876d7b3acbf96efb1b0366fe989a790e346dc977c52d0d4b682",
    }
    for name, digest in expected.items():
        if records[name]["sha256"] != digest:
            raise PreflightError(f"safe historical artifact hash mismatch: {name}")
    return records


def _proc_read(path: Path, limit: int = 1024 * 1024) -> bytes:
    fd = os.open(path, os.O_RDONLY)
    try:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(65536, limit - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise RuntimeError(f"proc read exceeds limit: {path}")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _ancestor_pids() -> set[int]:
    result = {os.getpid()}
    pid = os.getppid()
    while pid > 1 and pid not in result:
        result.add(pid)
        try:
            fields = _proc_read(Path(f"/proc/{pid}/stat"), 8192).decode("utf-8", "replace").split()
            pid = int(fields[3])
        except Exception:
            break
    return result


def collect_process_snapshot() -> dict[str, Any]:
    excluded = _ancestor_pids()
    relevant: list[dict[str, Any]] = []
    unknown_relevant: list[int] = []
    roots = [
        str(REPO_ROOT),
        str(CONTROL_ROOT),
        str(CACHE_ROOT),
        str(CHECKPOINT_ROOT),
        str(CHECKPOINT),
        *(str(path) for path in HISTORICAL_DENY_WRITE_ROOTS),
    ]
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit() or int(proc_dir.name) in excluded:
            continue
        pid = int(proc_dir.name)
        try:
            status = proc_dir.stat()
            cmd_raw = _proc_read(proc_dir / "cmdline", 1024 * 1024)
            argv_hash = bytes_sha256(cmd_raw)
            cmd_lower = cmd_raw.replace(b"\0", b" ").lower()
            argv_tokens = [token.lower() for token in cmd_raw.split(b"\0") if token]
            cwd = os.readlink(proc_dir / "cwd")
            exe = os.readlink(proc_dir / "exe")
            markers = [
                marker
                for marker in (b"run_c28", b"c28c", b"c28e", b"c28f")
                if marker in cmd_lower
            ]
            known_c28_runner = any(
                re.fullmatch(rb"run_c28[a-z0-9_.-]*\.py", token.rsplit(b"/", 1)[-1]) is not None
                for token in argv_tokens[:4]
            )
            writable_root_fds = 0
            fd_dir = proc_dir / "fd"
            for fd_entry in fd_dir.iterdir():
                try:
                    target = os.readlink(fd_entry)
                    if any(target == root or target.startswith(root.rstrip("/") + "/") for root in roots):
                        info = _proc_read(proc_dir / "fdinfo" / fd_entry.name, 16384).decode(
                            "utf-8", "replace"
                        )
                        flag_line = next((line for line in info.splitlines() if line.startswith("flags:")), "")
                        flags = int(flag_line.split()[1], 8) if flag_line else 0
                        if flags & (os.O_WRONLY | os.O_RDWR):
                            writable_root_fds += 1
                except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
                    continue
            if markers or writable_root_fds or cwd.startswith(str(REPO_ROOT)):
                start_fields = _proc_read(proc_dir / "stat", 8192).decode("utf-8", "replace").split()
                relevant.append(
                    {
                        "pid": pid,
                        "uid": int(status.st_uid),
                        "exe": exe,
                        "cwd": cwd,
                        "argv_sha256": argv_hash,
                        "markers": [item.decode("ascii") for item in markers],
                        "process_start_ticks": int(start_fields[21]),
                        "writable_c28_root_fd_count": writable_root_fds,
                        "known_c28_runner": known_c28_runner,
                    }
                )
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # Other-user processes are relevant only if they can be shown to own C28 roots; unknown is retained.
            unknown_relevant.append(pid)
    writer_conflicts = [
        item
        for item in relevant
        if item["writable_c28_root_fd_count"] > 0 or item["known_c28_runner"]
    ]

    gpu = _run_result(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu_apps = _run_result(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu_status = "AVAILABLE" if gpu["returncode"] == 0 and not gpu["timed_out"] else "UNKNOWN"
    gpu_rows = (
        [line for line in bytes(gpu["stdout"]).decode("utf-8", "replace").splitlines() if line]
        if gpu_status == "AVAILABLE"
        else []
    )
    app_rows = (
        [line for line in bytes(gpu_apps["stdout"]).decode("utf-8", "replace").splitlines() if line]
        if gpu_apps["returncode"] == 0 and not gpu_apps["timed_out"]
        else []
    )
    tmux = _run_result(["tmux", "list-sessions"])
    return {
        "relevant_processes": relevant,
        "writer_conflicts": writer_conflicts,
        "unknown_permission_process_count": len(unknown_relevant),
        "gpu_status": gpu_status,
        "gpu_compute_status": (
            "AVAILABLE"
            if gpu_apps["returncode"] == 0 and not gpu_apps["timed_out"]
            else "UNKNOWN"
        ),
        "gpu_rows": gpu_rows,
        "gpu_compute_rows": app_rows,
        "tmux_status": "AVAILABLE" if tmux["returncode"] == 0 else "NONE_OR_UNAVAILABLE",
        "tmux_stdout_sha256": bytes_sha256(bytes(tmux["stdout"])),
        "gpu_busy_blocks_only": ["G3_F0_A", "G5_F0_B"],
        "gpu0_cleanup_deferred_until_model_forward": True,
    }


def collect_static_review_receipt(protected_records: list[dict[str, Any]]) -> dict[str, Any]:
    _assert_content_source_not_protected(STATIC_REVIEW_RECEIPT, protected_records)
    receipt = load_json(STATIC_REVIEW_RECEIPT)
    expected_keys = {
        "schema_version",
        "status",
        "review_round",
        "reviewers",
        "reviewed_files",
        "authority_hashes",
        "constraints",
        "completed_at",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise PreflightError("static review receipt has an invalid exact schema")
    reviewers = receipt["reviewers"]
    reviewer_schema_invalid = (
        not isinstance(reviewers, list)
        or any(
            not isinstance(item, dict)
            or set(item) != {"task", "verdict"}
            or not isinstance(item["task"], str)
            or not item["task"].strip()
            or item["verdict"] != "STATIC_GO"
            for item in reviewers
        )
    )
    if (
        receipt["schema_version"] != "c28f_g0_static_review_receipt_v1"
        or receipt["status"] != "STATIC_GO"
        or type(receipt["review_round"]) is not int
        or receipt["review_round"] < 2
        or not isinstance(receipt["completed_at"], str)
        or not receipt["completed_at"].strip()
        or reviewer_schema_invalid
        or len({item["task"] for item in reviewers}) < 2
    ):
        raise PreflightError("static review receipt lacks two independent STATIC_GO verdicts")
    if receipt["authority_hashes"] != {
        "goal_contract": EXPECTED_GOAL_CONTRACT_SHA256,
        "blueprint": EXPECTED_BLUEPRINT_SHA256,
        "handoff": EXPECTED_HANDOFF_SHA256,
    }:
        raise PreflightError("static review receipt authority hashes mismatch")
    if receipt["constraints"] != {
        "code_executed_or_imported": False,
        "code_compiled_or_tested": False,
        "protected_content_opened": False,
        "historical_prediction_or_result_content_opened": False,
    }:
        raise PreflightError("static review receipt constraints mismatch")
    expected_records: dict[str, str] = {}
    for path in G0_REVIEWED_FILES:
        _assert_content_source_not_protected(path, protected_records)
        expected_records[path.relative_to(REPO_ROOT).as_posix()] = str(
            regular_file_record(path, content_hash=True)["sha256"]
        )
    if receipt["reviewed_files"] != expected_records:
        raise PreflightError("static review receipt does not match the current G0 code/fixture bytes")
    return {
        "payload": receipt,
        "file": regular_file_record(STATIC_REVIEW_RECEIPT, content_hash=True),
    }


def collect_immutable_lock(
    authority_files: dict[str, Any],
    protected_metadata_records: list[dict[str, Any]],
) -> dict[str, Any]:
    corpus = collect_corpus_and_splits(protected_metadata_records)
    protected = collect_protected_registry(corpus, protected_metadata_records)
    canonical = validate_canonical_paths(protected)
    namespace = validate_namespace_ownership(allow_partial_control=True)
    if namespace["recovery_temps"]:
        raise PreflightError(f"unrecovered atomic temp files: {namespace['recovery_temps']}")
    for path in BOOTSTRAP_FIXTURES.values():
        _assert_content_source_not_protected(path, protected_metadata_records)
    fixtures = {name: regular_file_record(path, content_hash=True) for name, path in BOOTSTRAP_FIXTURES.items()}
    _assert_content_source_not_protected(BOOTSTRAP_LOCK, protected_metadata_records)
    bootstrap_lock_bytes, bootstrap_lock_record = read_regular_bytes(
        BOOTSTRAP_LOCK, max_bytes=256, require_nlink_one=True
    )
    if bootstrap_lock_bytes != BOOTSTRAP_LOCK_HEADER:
        raise PreflightError("outer bootstrap lock header mismatch")
    lock = {
        "authority_files": authority_files,
        "checkpoint": lock_checkpoint(protected_metadata_records),
        "corpus_and_splits": corpus,
        "protected_registry": protected,
        "canonical_paths": canonical,
        "historical_artifacts": collect_historical_artifacts(protected),
        "legacy_source_hashes": collect_legacy_source_hashes(protected_metadata_records),
        "fixtures": fixtures,
        "static_review_receipt": collect_static_review_receipt(protected_metadata_records),
        "bootstrap_lock": bootstrap_lock_record,
        "quarantined_g0_attempts": collect_quarantined_g0_attempts(protected_metadata_records),
    }
    return {**lock, "immutable_lock_sha256": bytes_sha256(canonical_json_bytes(lock))}


def validate_bootstrap_control_parent() -> None:
    assert_allowed_write_path(CONTROL_ROOT / "single_writer.lock")
    writable_targets = (CODE_ROOT, TEST_ROOT, ENTRYPOINT, REPORT_ROOT, CACHE_ROOT, CHECKPOINT_ROOT)
    repo_resolved = REPO_ROOT.resolve(strict=True)
    repo_device = REPO_ROOT.stat().st_dev
    mount_lines = _proc_read(Path("/proc/self/mountinfo"), 16 * 1024 * 1024).decode(
        "utf-8", "strict"
    ).splitlines()

    def mount_unescape(value: str) -> str:
        return (
            value.replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\012", "\n")
            .replace("\\134", "\\")
        )

    mount_points = [Path(mount_unescape(line.split()[4])) for line in mount_lines if len(line.split()) > 5]
    for target in writable_targets:
        if not target.is_absolute() or ".." in target.parts:
            raise PreflightError(f"non-canonical bootstrap write target: {target}")
        _assert_path_chain_no_symlink(target)
        resolved = target.resolve(strict=False)
        if not _contains(repo_resolved, resolved):
            raise PreflightError(f"bootstrap write target escapes repository: {target}")
        existing = target
        while not os.path.lexists(existing):
            existing = existing.parent
        identity = lstat_identity(existing)
        if identity["type"] == "symlink" or int(identity["device"]) != int(repo_device):
            raise PreflightError(f"bootstrap target has a foreign/symlink ancestor: {target}")
        for mount_point in mount_points:
            mount_resolved = mount_point.resolve(strict=False)
            nested_repo_mount = mount_resolved != repo_resolved and _contains(
                repo_resolved, mount_resolved
            )
            if nested_repo_mount and (
                _contains(mount_resolved, resolved) or _contains(resolved, mount_resolved)
            ):
                raise PreflightError(
                    f"nested/bind mount overlaps bootstrap write target: {target} vs {mount_point}"
                )
    for path in HISTORICAL_DENY_WRITE_ROOTS + (PROTECTED_VAL_JSONL, PROTECTED_HISTORICAL_C28C_RESULTS):
        if not os.path.lexists(path):
            continue
        forbidden = path.resolve(strict=True)
        for target in writable_targets:
            resolved = target.resolve(strict=False)
            if _contains(forbidden, resolved) or _contains(resolved, forbidden):
                raise PreflightError(f"bootstrap target overlaps forbidden path: {target} vs {path}")
            if os.path.lexists(target) and os.path.samefile(target, path):
                raise PreflightError(f"bootstrap target aliases forbidden path: {target} vs {path}")


def _incident_bundle() -> dict[str, Any]:
    if not PRE_G0_SECURITY_INCIDENT_USER_ACK:
        raise PreflightError("SECURITY_INCIDENT_AWAITING_USER_ACK")
    bundle = {
        "incident": PRE_G0_SECURITY_INCIDENT,
        "impact_assessment": PRE_G0_SECURITY_INCIDENT_IMPACT_ASSESSMENT,
        "user_acknowledgement": {
            "text": PRE_G0_SECURITY_INCIDENT_USER_ACK_TEXT,
            "text_sha256": bytes_sha256(PRE_G0_SECURITY_INCIDENT_USER_ACK_TEXT.encode("utf-8")),
            "scope": "continue only because scientific impact is NONE; retain permanent audit exception",
        },
    }
    return {**bundle, "bundle_sha256": bytes_sha256(canonical_json_bytes(bundle))}


def _intent_file_bytes(intent: dict[str, Any]) -> bytes:
    return canonical_json_bytes(intent) + b"\n"


def _validate_intent_contract(intent: Any) -> dict[str, Any]:
    if not isinstance(intent, dict):
        raise PreflightError("G0 intent must be a JSON object")
    expected_keys = {
        "schema_version",
        "goal_id",
        "goal_objective",
        "attempt_id",
        "transaction_id",
        "created_at",
        "created_at_ns",
        "producer_argv",
        "producer_raw_argv",
        "producer_argv_sha256",
        "producer_raw_argv_sha256",
        "writer_lease",
        "incident_bundle",
        "immutable_lock",
        "git_baseline",
        "bootstrap_business_inventory",
        "process_snapshot",
        "disk",
        "intent_sha256",
    }
    if set(intent) != expected_keys:
        raise PreflightError(
            f"G0 intent keyset mismatch: missing={sorted(expected_keys - set(intent))} "
            f"extra={sorted(set(intent) - expected_keys)}"
        )
    if (
        intent["schema_version"] != "c28f_g0_bootstrap_intent_v3"
        or intent["goal_id"] != GOAL_ID
        or intent["goal_objective"] != GOAL_OBJECTIVE
    ):
        raise PreflightError("G0 intent schema/goal/objective mismatch")
    if (
        not isinstance(intent["attempt_id"], str)
        or not isinstance(intent["transaction_id"], str)
        or re.fullmatch(r"G0-A3-[0-9a-f]{32}", intent["attempt_id"]) is None
        or re.fullmatch(r"G0-TXN-[0-9a-f]{32}", intent["transaction_id"]) is None
    ):
        raise PreflightError("G0 intent attempt/transaction ID is non-canonical")
    if type(intent["created_at_ns"]) is not int or not isinstance(intent["created_at"], str):
        raise PreflightError("G0 intent timestamp fields are invalid")
    producer = intent["producer_argv"]
    raw_producer = intent["producer_raw_argv"]
    if (
        not isinstance(producer, list)
        or not all(isinstance(item, str) for item in producer)
        or len(producer) != 5
        or producer[1:] != ["-B", str(ENTRYPOINT), "--action", "G0_INIT"]
        or not isinstance(raw_producer, list)
        or not all(isinstance(item, str) for item in raw_producer)
        or len(raw_producer) != 5
        or raw_producer[1] != "-B"
        or raw_producer[-2:] != ["--action", "G0_INIT"]
    ):
        raise PreflightError("G0 intent producer argv shape is invalid")
    if (
        Path(producer[0]).resolve(strict=True) != Path(producer[0])
        or Path(raw_producer[0]).resolve(strict=True) != Path(producer[0])
        or Path(raw_producer[2]).resolve(strict=True) != ENTRYPOINT
    ):
        raise PreflightError("G0 intent producer paths are not canonical")
    if intent["producer_argv_sha256"] != bytes_sha256(canonical_json_bytes(producer)) or intent[
        "producer_raw_argv_sha256"
    ] != bytes_sha256(canonical_json_bytes(raw_producer)):
        raise PreflightError("G0 intent producer argv hash mismatch")
    writer_lease = intent["writer_lease"]
    if (
        not isinstance(writer_lease, dict)
        or set(writer_lease)
        != {"path", "device", "inode", "mode", "nlink", "header_sha256", "lease_lineage_root_id"}
        or writer_lease["path"] != str(CONTROL_ROOT / "single_writer.lock")
        or type(writer_lease["device"]) is not int
        or type(writer_lease["inode"]) is not int
        or writer_lease["nlink"] != 1
        or writer_lease["header_sha256"] != bytes_sha256(b"C28F_SINGLE_WRITER_LOCK_V2\n")
        or not isinstance(writer_lease["lease_lineage_root_id"], str)
        or re.fullmatch(r"[0-9a-f]{32}", writer_lease["lease_lineage_root_id"]) is None
    ):
        raise PreflightError("G0 intent writer lease binding is invalid")
    if intent["incident_bundle"] != _incident_bundle():
        raise PreflightError("G0 intent incident/ack bundle mismatch")
    git = intent["git_baseline"]
    if not isinstance(git, dict) or git.get("branch") != EXPECTED_BRANCH or git.get("head") != EXPECTED_HEAD:
        raise PreflightError("G0 intent branch/HEAD baseline mismatch")
    preexisting = git.get("preexisting_dirty_inventory")
    if not isinstance(preexisting, dict) or git.get("preexisting_dirty_inventory_sha256") != bytes_sha256(
        canonical_json_bytes(preexisting)
    ):
        raise PreflightError("G0 intent dirty inventory self-hash mismatch")
    immutable = intent["immutable_lock"]
    if not isinstance(immutable, dict) or "immutable_lock_sha256" not in immutable:
        raise PreflightError("G0 intent immutable lock is invalid")
    immutable_payload = {key: value for key, value in immutable.items() if key != "immutable_lock_sha256"}
    if immutable["immutable_lock_sha256"] != bytes_sha256(canonical_json_bytes(immutable_payload)):
        raise PreflightError("G0 intent immutable lock self-hash mismatch")
    business = intent["bootstrap_business_inventory"]
    if not isinstance(business, dict):
        raise PreflightError("G0 intent business inventory is invalid")
    business_domain = {"schema": business.get("schema"), "entries": business.get("entries")}
    if business.get("business_delta_sha256") != bytes_sha256(canonical_json_bytes(business_domain)):
        raise PreflightError("G0 intent business inventory self-hash mismatch")
    quarantined = immutable.get("quarantined_g0_attempts")
    if not isinstance(quarantined, dict) or set(quarantined) != set(QUARANTINED_G0_ATTEMPTS):
        raise PreflightError("G0 intent quarantine lineage set mismatch")
    _assert_quarantine_business_closure(business, quarantined)
    expected_intent_sha = semantic_sha256(intent, excluded_fields=("intent_sha256",))
    if intent["intent_sha256"] != expected_intent_sha:
        raise PreflightError("G0 intent semantic hash mismatch")
    return intent


def _load_and_verify_intent(
    path: Path,
    root_path: Path,
    *,
    require_detached_root: bool = True,
) -> dict[str, Any]:
    intent = load_json(path)
    expected_semantic = semantic_sha256(intent, excluded_fields=("intent_sha256",))
    if intent.get("intent_sha256") != expected_semantic:
        raise PreflightError("G0 intent semantic hash mismatch")
    data, record = read_regular_bytes(path, require_nlink_one=True)
    if os.path.lexists(root_path):
        root_bytes, _root_record = read_regular_bytes(root_path, max_bytes=256, require_nlink_one=True)
        if root_bytes != (record["sha256"] + "\n").encode("ascii"):
            raise PreflightError("G0 intent detached root mismatch")
    elif require_detached_root:
        raise PreflightError("G0 intent detached root is missing")
    return _validate_intent_contract(intent)


def _build_authority_ledger(intent: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "goal_id": GOAL_ID,
        "created_at": intent["created_at"],
        "permissions": {
            "AUTH_F0_F1_IMPLEMENTATION": {"state": "ACTIVE", "value": True, "source": "explicit_goal_start"},
            "AUTH_PROTECTED_ID_MAPPING_ONLY": {
                "state": "ACTIVE",
                "value": True,
                "scope": "desc_id_to_video_id_only_no_text_timestamp_feature_label_metric",
                "source": "goal_launch_contract_executed_by_user",
            },
            "AUTH_F2_F5_PROTOTYPE": {"state": "UNGRANTED", "value": False},
            "AUTH_F6_MECHANISM": {"state": "UNGRANTED", "value": False},
            "AUTH_F8_FORMAL": {"state": "UNGRANTED", "value": False},
            "AUTH_HOLDOUT_OFFICIAL": {"state": "UNGRANTED", "value": False},
            "AUTH_GIT_BRANCH_CREATE_SWITCH": {"state": "UNGRANTED", "value": False},
            "AUTH_GIT_COMMIT": {"state": "UNGRANTED", "value": False},
            "AUTH_GIT_PUSH_PR": {"state": "UNGRANTED", "value": False},
        },
        "formal_data_policy": "STRICT_CORE_ONLY",
        "u_formal_updates": 17_360,
        "real_data_training_authorized": False,
        "protected_model_evaluation_authorized": False,
        "pre_g0_security_incident": intent["incident_bundle"],
        "continuation_disposition": "ACKNOWLEDGED_CONTINUE_WITH_RECORDED_EXCEPTION_NEW_ATTEMPT",
    }


def _build_execution_scope(intent: dict[str, Any]) -> dict[str, Any]:
    git = intent["git_baseline"]
    business = intent["bootstrap_business_inventory"]
    return {
        "schema_version": SCHEMA_VERSION,
        "goal_id": GOAL_ID,
        "objective": GOAL_OBJECTIVE,
        "expected_branch": EXPECTED_BRANCH,
        "expected_head": EXPECTED_HEAD,
        "immutable_lock_sha256": intent["immutable_lock"]["immutable_lock_sha256"],
        "static_review_receipt": intent["immutable_lock"]["static_review_receipt"],
        "canonical_roots": intent["immutable_lock"]["canonical_paths"],
        "allowed_write_globs": list(ALLOWED_WRITE_GLOBS),
        "immutable_deny_write_roots": [str(QUARANTINE_ROOT)]
        + [str(path) for path in HISTORICAL_DENY_WRITE_ROOTS],
        "outer_bootstrap_lock": {
            "path": str(BOOTSTRAP_LOCK),
            "header_sha256": bytes_sha256(BOOTSTRAP_LOCK_HEADER),
            "locked_record": intent["immutable_lock"]["bootstrap_lock"],
            "scope": "required for quarantine migration and every live G0 action",
            "business_delta_excluded": True,
        },
        "shared_legacy_files_writable": False,
        "shared_write_exception": None,
        "hash_domains": {
            "business_delta": (
                "sorted Goal-owned regular path+size+full SHA; only the live "
                "reports/c28f_v5/goal_control tree and fixed bootstrap.lock capability are excluded; "
                "the quarantine subtree is fully included"
            ),
            "goal_state": "canonical JSON excluding state_sha256",
            "goal_event": "canonical JSON excluding event_sha256, with chained prev_event_sha256",
            "artifact_manifest": "manifest excludes itself and detached root",
            "intent": "canonical JSON excluding intent_sha256 plus detached file-byte root",
        },
        "bootstrap_order_refinement": {
            "contract_text": "lease -> initial registries -> baseline/state -> GENESIS -> fsync",
            "implemented_order": (
                "lease -> preliminary in-memory snapshot for recovery planning -> fsynced self-hashed intent -> "
                "initial registries -> authoritative business snapshot and state recalculation -> "
                "GENESIS+fsync -> projections -> GOAL_STATE commit marker"
            ),
            "reason": (
                "the preliminary snapshot makes registry bytes recoverable; the authoritative post-registry "
                "recalculation satisfies the commit order without placeholder registries"
            ),
            "business_hash_equivalence": (
                "only the live goal_control tree and fixed outer lock are excluded; every quarantined "
                "failed-attempt file and migration receipt is included in business_delta_sha256"
            ),
            "scientific_decision_impact": "NONE",
            "acceptance_condition": "bound independent static-review receipt must be STATIC_GO",
        },
        "preexisting_dirty_inventory_sha256": git["preexisting_dirty_inventory_sha256"],
        "goal_owned_path_allowlist": list(GOAL_OWNED_REPO_PREFIXES)
        + list(GOAL_OWNED_EXACT_REPO_PATHS),
        "expected_goal_delta_sha256": business["business_delta_sha256"],
        "unexpected_external_delta": [],
        "unexpected_external_delta_sha256": bytes_sha256(canonical_json_bytes([])),
        "bootstrap_inputs": {
            "classification": "OBSERVED_PRE_GENESIS_INPUT_REQUIRES_BOUND_STATIC_REVIEW_RECEIPT",
            "business_delta_sha256": business["business_delta_sha256"],
            "exception_rationale": (
                "Minimal control code/static fixtures existed before GENESIS because no G0 writer existed; "
                "their exact hashes are bound here, and the launcher may execute only after a matching "
                "independent static-review receipt is added to the business inventory."
            ),
        },
        "quarantined_g0_attempts": intent["immutable_lock"]["quarantined_g0_attempts"],
    }


def _build_evidence_lock(intent: dict[str, Any]) -> dict[str, Any]:
    immutable = intent["immutable_lock"]
    incident = intent["incident_bundle"]
    return {
        "schema_version": "c28f_f0_evidence_lock_v2",
        "evidence_lock_id": f"C28F-F0-A3-{EXPECTED_CHECKPOINT_SHA256[:16]}",
        "created_at": intent["created_at"],
        "authority_files": immutable["authority_files"],
        "repo": {
            "path": str(REPO_ROOT),
            "branch": intent["git_baseline"]["branch"],
            "head": intent["git_baseline"]["head"],
            "tracked_diff_sha256": intent["git_baseline"]["tracked_diff_sha256"],
            "staged_diff_sha256": intent["git_baseline"]["staged_diff_sha256"],
        },
        "checkpoint": immutable["checkpoint"],
        "corpus_and_splits": immutable["corpus_and_splits"],
        "protected_registry": immutable["protected_registry"],
        "process_snapshot": intent["process_snapshot"],
        "pre_attempt_cumulative_access": {
            "incident_count": 1,
            "prediction_values_surfaced": incident["incident"]["prediction_values_surfaced"],
            "model_score_values_surfaced": incident["incident"]["model_score_values_surfaced"],
            "label_or_timestamp_values_surfaced": incident["incident"][
                "label_or_timestamp_values_surfaced"
            ],
            "model_evaluation_count": 0,
        },
        "current_clean_attempt_access": {
            "protected_sample_content_reads": 0,
            "protected_prediction_content_reads": 0,
            "protected_metric_content_reads": 0,
            "protected_model_evaluations": 0,
            "model_forward_evaluations": 0,
            "optimizer_updates": 0,
        },
        "protected_desc_id_manifest_registry": immutable["corpus_and_splits"][
            "protected_desc_id_manifest_registry"
        ],
        "quarantined_g0_attempts": immutable["quarantined_g0_attempts"],
    }


def _build_baseline_registry(intent: dict[str, Any]) -> dict[str, Any]:
    immutable = intent["immutable_lock"]
    parents = {
        "G1": "PREFLIGHT_OK",
        "G2": "FIREWALL_OK",
        "G3": "METRIC_CONTRACT_OK",
        "G4": "F0A_COMPLETE",
        "G5": "ROLE_POLICY_LOCKED",
        "G6": "F0B_COMPLETE",
        "G7": "F1_COMPLETE",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": intent["created_at"],
        "authority_files": immutable["authority_files"],
        "repo": {
            "branch": intent["git_baseline"]["branch"],
            "head": intent["git_baseline"]["head"],
            "preexisting_dirty_inventory_sha256": intent["git_baseline"][
                "preexisting_dirty_inventory_sha256"
            ],
            "expected_goal_delta_sha256": intent["bootstrap_business_inventory"][
                "business_delta_sha256"
            ],
        },
        "checkpoint": immutable["checkpoint"],
        "corpus_and_splits": immutable["corpus_and_splits"],
        "legacy_source_hashes": immutable["legacy_source_hashes"],
        "historical_artifacts": immutable["historical_artifacts"],
        "protected_historical_c28c_result": {
            "metadata_only": _metadata_or_missing(PROTECTED_HISTORICAL_C28C_RESULTS),
            "content_hash_read_in_current_attempt": False,
            "replay_allowed": False,
        },
        "quarantined_g0_attempts": immutable["quarantined_g0_attempts"],
        "c28f_metric_schema": "PENDING_G2",
        "feature_manifest_scope": "source identity only; derived bank full SHA required before F0-A",
        "last_green": {
            "experiment": "C28E_3_FROZEN_CHECKPOINT_AND_NONPROTECTED_CALIB_SELECT_EVIDENCE",
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "decision_artifact_sha256": immutable["historical_artifacts"]["c28e_decision"]["sha256"],
            "training_log_sha256": immutable["historical_artifacts"]["c28e_training_log"]["sha256"],
        },
        "stage_lineage": {
            gate: {
                "parent_state": parent,
                "status": "PENDING_NOT_IMPLEMENTED",
                "planned_action_identifier": {
                    "G1": "G1_FAIL_CLOSED_GUARD",
                    "G2": "G2_METRIC_CONTRACT",
                    "G3": "G3_F0_A",
                    "G4": "G4_ROLE_POLICY_LOCK",
                    "G5": "G5_F0_B",
                    "G6": "G6_F1_CONTRACTS",
                    "G7": "G7_FINALIZE",
                }[gate],
                "replay_command_argv_sha256": None,
                "replay_command_status": "MUST_BE_FROZEN_WHEN_THE_REVIEWED_ACTION_EXISTS",
            }
            for gate, parent in parents.items()
        },
    }


def _build_gate_registry(intent: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": intent["created_at"],
        "unit": "ratio",
        "gates": {
            "G0": {"status": "PASS", "evidence": "evidence_lock.json"},
            "G1": {"status": "PENDING", "next_action": "G1_FAIL_CLOSED_GUARD"},
            "G2": {"status": "PENDING"},
            "G3": {"status": "PENDING", "logical_eval_cap": 1},
            "G4": {"status": "PENDING"},
            "G5": {"status": "PENDING", "logical_eval_cap": 1},
            "G6": {"status": "PENDING"},
            "G7": {"status": "PENDING"},
        },
        "numeric_contract": {
            "G_broad_min": 0.90,
            "G_keep_min": 0.90,
            "G_front_min": 0.80,
            "G_late_identity_tolerance": 1e-8,
            "query_completeness": 1.0,
            "historical_evaluator_parity_abs_max": 1e-6,
        },
        "missing_semantics": "FAIL_CLOSED unless explicitly schema-optional",
        "applicability": {
            "F0_F1_safety_metric_schedule_temporal_recovery": "APPLICABLE",
            "F2_F5_candidate_global_loss_gradient_pressure": "NOT_APPLICABLE_SCHEMA_ONLY",
            "real_data_training": "NOT_AUTHORIZED",
        },
    }


def _build_budget(intent: dict[str, Any]) -> dict[str, Any]:
    immutable = intent["immutable_lock"]
    split = immutable["corpus_and_splits"]["split_manifests"]
    historical = immutable["historical_artifacts"]
    fixtures = immutable["fixtures"]
    token = lambda purpose: bytes_sha256(f"{GOAL_ID}:{intent['attempt_id']}:{purpose}:token-1".encode())
    replay_names = (
        "c7_train_calib_predictions",
        "c7_train_calib_scores",
        "c7_train_calib_cache",
    )
    fixture_names = ("metric_fixture_v1", "state_fixture_v1")
    return {
        "schema_version": "c28f_budget_v2",
        "created_at": intent["created_at"],
        "caps": {
            "F0_A_logical_model_forward_eval": 1,
            "F0_B_logical_model_forward_eval": 1,
            "protected_eval": 0,
            "real_data_optimizer_updates": 0,
            "synthetic_fixture_optimizer_updates": 64,
            "frozen_artifact_protocol_replay_attempts": 32,
            "real_data_training_arms_or_seeds": 0,
            "new_performance_based_thresholds": 0,
            "same_root_cause_retries": 2,
            "concurrent_writers": 1,
        },
        "usage": {
            "F0_A_logical_model_forward_eval": 0,
            "F0_B_logical_model_forward_eval": 0,
            "protected_eval": 0,
            "real_data_optimizer_updates": 0,
            "synthetic_fixture_optimizer_updates": 0,
            "frozen_artifact_protocol_replay_attempts": 0,
            "same_root_cause_retries": 1,
        },
        "bootstrap_repair_ledger": [
            {
                "root_cause": immutable["quarantined_g0_attempts"][attempt_id]["root_cause"],
                "quarantined_attempt_id": attempt_id,
                "quarantined_transaction_id": immutable["quarantined_g0_attempts"][attempt_id][
                    "transaction_id"
                ],
                "control_tree_sha256": immutable["quarantined_g0_attempts"][attempt_id][
                    "control_tree_sha256"
                ],
                "same_root_cause_retries_used": 1,
                "same_root_cause_retries_remaining": 1,
                "model_forward_evaluations": immutable["quarantined_g0_attempts"][attempt_id][
                    "model_forward_evaluations"
                ],
                "scientific_impact": immutable["quarantined_g0_attempts"][attempt_id][
                    "scientific_impact"
                ],
            }
            for attempt_id in sorted(immutable["quarantined_g0_attempts"])
        ],
        "synthetic_fixture_allowlist": [
            {
                "name": name,
                "path": fixtures[name]["path"],
                "sha256": fixtures[name]["sha256"],
            }
            for name in fixture_names
        ],
        "artifact_replay_allowlist": [
            {
                "name": name,
                "path": historical[name]["path"],
                "sha256": historical[name]["sha256"],
            }
            for name in replay_names
        ],
        "eval_tokens": {
            "F0_A": {
                "token_id": token("F0_A"),
                "state": "UNISSUED",
                "purpose": "F0_A_TEACHER_FREE_CALIB_SELECT_FORENSICS",
                "expected_split_manifest_sha256": split["calib_select"]["sha256"],
                "logical_cap": 1,
            },
            "F0_B": {
                "token_id": token("F0_B"),
                "state": "UNISSUED",
                "purpose": "F0_B_TEACHER_FREE_ROUTE_DEV_REPLICATION",
                "expected_split_manifest_sha256": "PENDING_G4_ROLE_MANIFEST_LOCK",
                "logical_cap": 1,
            },
        },
        "token_state_machine": ["UNISSUED", "RESERVED", "RUNNING", "COMPLETED"],
        "teacher_candidate_eval_count": 0,
        "gt_support_eval_count": 0,
        "gt_append_eval_count": 0,
    }


def _build_rollback_map(intent: dict[str, Any]) -> dict[str, Any]:
    rows = {
        "G0_BOOTSTRAP": [
            "NO_COMMITTED_LAST_GREEN",
            "invalid control transaction preserved under reports/c28f_v5/quarantine",
            "new attempt_id after exact quarantine tree lock and full static re-review",
        ],
        "G1": ["PREFLIGHT_OK", "guard/test/report attempt", "new attempt; full safety+legacy replay"],
        "G2": ["FIREWALL_OK", "metric/schema/replay attempt", "new schema attempt; preserve legacy fields"],
        "G3_IO": ["METRIC_CONTRACT_OK+same_token", "incomplete chunks", "same eval_id/cursor/input hashes"],
        "G3_SEMANTIC": ["METRIC_CONTRACT_OK", "forward descendants; token consumed", "request new token"],
        "G4": ["F0A_COMPLETE", "unlocked role-policy attempt", "rebuild from same input hash"],
        "G5": ["ROLE_POLICY_LOCKED", "same token rules as G3", "same-token recovery only"],
        "G6": ["F0B_COMPLETE", "synthetic/scaffold attempt", "new attempt within synthetic cap"],
        "G7_PRE_INTENT": ["F1_COMPLETE", "uncommitted finalization", "new attempt, same artifacts"],
        "G7_POST_INTENT": ["PENDING_FINALIZATION", "no final artifact quarantine", "same txn only"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": intent["created_at"],
        "entries": {
            key: {"last_green": row[0], "quarantine_scope": row[1], "non_destructive_repair": row[2]}
            for key, row in rows.items()
        },
        "destructive_git_rollback_forbidden": True,
        "max_same_root_cause_retries": 2,
    }


def _build_artifact_manifest(intent: dict[str, Any]) -> dict[str, Any]:
    entries = []
    for item in intent["bootstrap_business_inventory"]["entries"]:
        quarantined = item["path"].startswith(
            "blueprint_e2e_v2/reports/c28f_v5/quarantine/"
        )
        migration_receipt = quarantined and item["path"].endswith("/MIGRATION_RECEIPT.json")
        quarantine_attempt_id = None
        quarantine_record = None
        if quarantined:
            parts = Path(item["path"]).parts
            quarantine_attempt_id = parts[parts.index("quarantine") + 1]
            quarantine_record = intent["immutable_lock"]["quarantined_g0_attempts"][
                quarantine_attempt_id
            ]
        entries.append(
            {
                **item,
                "producer_kind": (
                    "QUARANTINE_MIGRATION_RECEIPT"
                    if migration_receipt
                    else (
                        "PRESERVED_FAILED_G0_CONTROL_ATTEMPT"
                        if quarantined
                        else "OBSERVED_PRE_GENESIS_INPUT_BOUND_TO_STATIC_REVIEW_RECEIPT"
                    )
                ),
                "registrar_argv_sha256": intent["producer_argv_sha256"],
                "input_lineage": [EXPECTED_GOAL_CONTRACT_SHA256, EXPECTED_BLUEPRINT_SHA256, EXPECTED_HEAD],
                "split": None,
                "teacher_candidate": False,
                "gt_support": False,
                "gt_append": False,
                "real_data_optimizer_updates": 0,
                "scientific_valid": not quarantined,
                "replay_allowed": False,
                "quarantined_attempt_id": quarantine_attempt_id,
                "quarantined_transaction_id": (
                    quarantine_record["transaction_id"] if quarantine_record else None
                ),
                "quarantined_control_tree_sha256": (
                    quarantine_record["control_tree_sha256"] if quarantine_record else None
                ),
                "quarantine_root_cause_fingerprint": (
                    quarantine_record["failure_fingerprint"] if quarantine_record else None
                ),
                "model_forward_evaluations": (
                    quarantine_record["model_forward_evaluations"] if quarantine_record else 0
                ),
                "status": (
                    "VALID_QUARANTINE_AUDIT_RECEIPT"
                    if migration_receipt
                    else (
                        "QUARANTINED_INVALID_CONTROL_TRANSACTION"
                        if quarantined
                        else "VALID_BOOTSTRAP_INPUT"
                    )
                ),
            }
        )
    return {
        "schema_version": "c28f_artifact_manifest_v2",
        "created_at": intent["created_at"],
        "entries": entries,
        "self_excluded": True,
        "detached_root_excluded": True,
    }


def _canonical_file_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _build_registry_payloads(intent: dict[str, Any]) -> dict[str, Any]:
    return {
        "execution_scope.json": _build_execution_scope(intent),
        "authority_ledger.json": _build_authority_ledger(intent),
        "evidence_lock.json": _build_evidence_lock(intent),
        "baseline_registry.json": _build_baseline_registry(intent),
        "gate_registry.json": _build_gate_registry(intent),
        "budget.json": _build_budget(intent),
        "rollback_map.json": _build_rollback_map(intent),
        "preexisting_dirty_inventory.json": intent["git_baseline"]["preexisting_dirty_inventory"],
    }


def _continuation_text(state: dict[str, Any], event: dict[str, Any]) -> str:
    authority = state["authority_summary"]
    lines = [
        "# C28F v5 F0/F1 Goal Continuation",
        "",
        f"- Goal ID: `{state['goal_id']}`",
        f"- Attempt: `{state['attempt_id']}`",
        f"- State: `{state['status']}`",
        f"- Unique next action: `{state['next_action']}`",
        f"- State SHA256: `{state['state_sha256']}`",
        f"- Last event SHA256: `{event['event_sha256']}`",
        f"- Expected business delta SHA256: `{state['expected_goal_delta_sha256']}`",
        f"- F0/F1 authority: `{authority['AUTH_F0_F1_IMPLEMENTATION']}`",
        f"- Protected ID-only authority: `{authority['AUTH_PROTECTED_ID_MAPPING_ONLY']}`",
        "- Pre-G0 protected access incident remains permanently recorded; in this clean attempt, forbidden protected content access and authorized ID-manifest content opens are both 0.",
        "- One non-scientific G0 control transaction is preserved under the quarantine subtree after a canonical-order self-verification failure; it is not a valid baseline or decision artifact.",
        "- Higher-stage, protected-eval, training and Git-write authorities remain `UNGRANTED`.",
        "- Resume only after verifying intent, plan, event chain, state, authority, worktree and checkpoint locks.",
        "",
    ]
    return "\n".join(lines)


def _prepare_transaction(intent: dict[str, Any]) -> dict[str, Any]:
    registries = _build_registry_payloads(intent)
    registry_bytes = {name: _canonical_file_bytes(payload) for name, payload in registries.items()}
    artifact_manifest = _build_artifact_manifest(intent)
    artifact_bytes = _canonical_file_bytes(artifact_manifest)
    artifact_root_bytes = (bytes_sha256(artifact_bytes) + "\n").encode("ascii")
    authority = registries["authority_ledger.json"]
    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "goal_id": GOAL_ID,
        "attempt_id": intent["attempt_id"],
        "transaction_id": intent["transaction_id"],
        "objective_sha256": bytes_sha256(GOAL_OBJECTIVE.encode("utf-8")),
        "status": "PREFLIGHT_OK",
        "stage": "G0",
        "transition_seq": 0,
        "completed_gates": ["G0"],
        "next_action": "G1_FAIL_CLOSED_GUARD",
        "next_action_is_unique": True,
        "expected_branch": EXPECTED_BRANCH,
        "expected_head": EXPECTED_HEAD,
        "preexisting_dirty_inventory_sha256": intent["git_baseline"][
            "preexisting_dirty_inventory_sha256"
        ],
        "expected_goal_delta_sha256": intent["bootstrap_business_inventory"][
            "business_delta_sha256"
        ],
        "expected_external_delta_sha256": bytes_sha256(canonical_json_bytes([])),
        "unexpected_external_delta": [],
        "authority_summary": {
            key: value["state"] for key, value in authority["permissions"].items()
        },
        "eval_token_states": {"F0_A": "UNISSUED", "F0_B": "UNISSUED"},
        "real_data_optimizer_updates": 0,
        "protected_model_evals": 0,
        "pre_g0_incident_exception": intent["incident_bundle"]["bundle_sha256"],
        "updated_at": intent["created_at"],
    }
    state["state_sha256"] = semantic_sha256(state, excluded_fields=("state_sha256",))
    output_hashes = {name: bytes_sha256(data) for name, data in registry_bytes.items()}
    output_hashes.update(
        {
            "artifact_manifest.json": bytes_sha256(artifact_bytes),
            "artifact_manifest.root.sha256": bytes_sha256(artifact_root_bytes),
        }
    )
    plan_base = {
        "schema_version": "c28f_g0_transaction_plan_v1",
        "goal_id": GOAL_ID,
        "attempt_id": intent["attempt_id"],
        "transaction_id": intent["transaction_id"],
        "registry_output_hashes": output_hashes,
        "expected_state_sha256": state["state_sha256"],
        "expected_business_delta_sha256": state["expected_goal_delta_sha256"],
    }
    plan = {**plan_base, "plan_sha256": bytes_sha256(canonical_json_bytes(plan_base))}
    plan_bytes = _canonical_file_bytes(plan)
    event: dict[str, Any] = {
        "schema_version": "c28f_goal_event_v1",
        "seq": 0,
        "timestamp": intent["created_at"],
        "action": "G0_SCOPE_LOCK_GENESIS_AFTER_ACKNOWLEDGED_INCIDENT_AND_QUARANTINED_A2",
        "attempt_id": intent["attempt_id"],
        "transaction_id": intent["transaction_id"],
        "input_hashes": {
            "goal_contract": EXPECTED_GOAL_CONTRACT_SHA256,
            "blueprint": EXPECTED_BLUEPRINT_SHA256,
            "handoff": EXPECTED_HANDOFF_SHA256,
            "checkpoint": EXPECTED_CHECKPOINT_SHA256,
            "intent_semantic": intent["intent_sha256"],
            "intent_file": bytes_sha256(_intent_file_bytes(intent)),
            "transaction_plan_file": bytes_sha256(plan_bytes),
            "incident_bundle": intent["incident_bundle"]["bundle_sha256"],
            "writer_lease": bytes_sha256(canonical_json_bytes(intent["writer_lease"])),
            "preexisting_dirty_inventory": intent["git_baseline"][
                "preexisting_dirty_inventory_sha256"
            ],
        },
        "output_hashes": output_hashes,
        "result": "PASS_WITH_ACKNOWLEDGED_PRE_G0_AUDIT_EXCEPTION",
        "prev_event_sha256": None,
        "expected_next_state_sha256": state["state_sha256"],
        "expected_goal_delta_sha256": state["expected_goal_delta_sha256"],
        "expected_external_delta_sha256": bytes_sha256(canonical_json_bytes([])),
        "next_action": "G1_FAIL_CLOSED_GUARD",
    }
    event["event_sha256"] = semantic_sha256(event, excluded_fields=("event_sha256",))
    continuation = _continuation_text(state, event).encode("utf-8")
    return {
        "registries": registries,
        "registry_bytes": registry_bytes,
        "artifact_manifest": artifact_manifest,
        "artifact_bytes": artifact_bytes,
        "artifact_root_bytes": artifact_root_bytes,
        "state": state,
        "state_bytes": _canonical_file_bytes(state),
        "plan": plan,
        "plan_bytes": plan_bytes,
        "event": event,
        "event_bytes": _canonical_file_bytes(event),
        "continuation": continuation,
    }


def _assert_transaction_canonical_roundtrip(
    intent: dict[str, Any], txn: dict[str, Any]
) -> None:
    reloaded = json.loads(canonical_json_bytes(intent).decode("utf-8"))
    replay = _prepare_transaction(reloaded)
    if replay != txn:
        raise PreflightError(
            "transaction builder depends on pre-serialization mapping order or non-canonical state"
        )


def _write_intent(intent: dict[str, Any], *, lease: WriterLease) -> None:
    path = CONTROL_ROOT / "g0_bootstrap_intent.json"
    root = CONTROL_ROOT / "g0_bootstrap_intent.root.sha256"
    data = _intent_file_bytes(intent)
    record = atomic_write_immutable(path, data, lease=lease)
    atomic_write_immutable(root, (record["sha256"] + "\n").encode("ascii"), lease=lease)


def _g0_expected_namespace(
    txn: dict[str, Any], intent: dict[str, Any]
) -> tuple[set[str], set[str]]:
    top_level_files = {
        "g0_bootstrap_intent.json",
        "g0_bootstrap_intent.root.sha256",
        "execution_scope.json",
        "authority_ledger.json",
        "evidence_lock.json",
        "baseline_registry.json",
        "gate_registry.json",
        "budget.json",
        "rollback_map.json",
        "preexisting_dirty_inventory.json",
        "artifact_manifest.json",
        "artifact_manifest.root.sha256",
        "GOAL_EVENTS.jsonl",
        "GOAL_STATE.json",
        "CONTINUATION.md",
        "single_writer.lock",
        "single_writer.heartbeat.json",
    }
    nested_files = {
        f"transactions/{intent['transaction_id']}/plan.json",
        f"state_records/000000_{txn['state']['state_sha256']}.json",
        f"event_records/000000_{txn['event']['event_sha256']}.jsonl",
        f"registry_records/000000_artifact_manifest_{bytes_sha256(txn['artifact_bytes'])}.json",
    }
    nested_files.update(
        f"registry_records/000000_{name.replace('.json', '')}_{bytes_sha256(data)}.json"
        for name, data in txn["registry_bytes"].items()
    )
    directories = {
        "transactions",
        f"transactions/{intent['transaction_id']}",
        "registry_records",
        "state_records",
        "event_records",
    }
    return top_level_files | nested_files, directories


def _assert_g0_recovery_namespace_subset(txn: dict[str, Any], intent: dict[str, Any]) -> None:
    namespace = validate_namespace_ownership(allow_partial_control=True)
    if namespace["recovery_temps"]:
        raise PreflightError(f"unresolved recovery temp files: {namespace['recovery_temps']}")
    expected_files, expected_directories = _g0_expected_namespace(txn, intent)
    actual_files = set(namespace["control_files"])
    actual_directories = set(namespace["control_directories"])
    if not actual_files.issubset(expected_files) or not actual_directories.issubset(
        expected_directories
    ):
        raise PreflightError(
            "partial G0 namespace contains files outside the intent closure: "
            f"files={sorted(actual_files - expected_files)} "
            f"dirs={sorted(actual_directories - expected_directories)}"
        )


def _install_transaction(
    txn: dict[str, Any],
    intent: dict[str, Any],
    *,
    lease: WriterLease,
    protected_identities: set[tuple[int, int]],
) -> None:
    txn_dir = CONTROL_ROOT / "transactions" / intent["transaction_id"]
    atomic_write_immutable(txn_dir / "plan.json", txn["plan_bytes"], lease=lease)
    for name, data in txn["registry_bytes"].items():
        digest = bytes_sha256(data)
        safe_name = name.replace(".json", "")
        atomic_write_immutable(
            CONTROL_ROOT / "registry_records" / f"000000_{safe_name}_{digest}.json",
            data,
            lease=lease,
        )
        atomic_write_bytes(
            CONTROL_ROOT / name,
            data,
            lease=lease,
            precondition="ABSENT_OR_IDENTICAL",
        )
    artifact_digest = bytes_sha256(txn["artifact_bytes"])
    atomic_write_immutable(
        CONTROL_ROOT / "registry_records" / f"000000_artifact_manifest_{artifact_digest}.json",
        txn["artifact_bytes"],
        lease=lease,
    )
    atomic_write_bytes(
        CONTROL_ROOT / "artifact_manifest.json",
        txn["artifact_bytes"],
        lease=lease,
        precondition="ABSENT_OR_IDENTICAL",
    )
    atomic_write_bytes(
        CONTROL_ROOT / "artifact_manifest.root.sha256",
        txn["artifact_root_bytes"],
        lease=lease,
        precondition="ABSENT_OR_IDENTICAL",
    )
    authoritative_business = business_inventory(forbidden_identities=protected_identities)
    if authoritative_business != intent["bootstrap_business_inventory"]:
        raise PreflightError("authoritative post-registry business baseline drift")
    authoritative_txn = _prepare_transaction(intent)
    if authoritative_txn != txn:
        raise PreflightError("post-registry full transaction recalculation mismatch")
    atomic_write_immutable(
        CONTROL_ROOT / "state_records" / f"000000_{txn['state']['state_sha256']}.json",
        txn["state_bytes"],
        lease=lease,
    )
    append_recoverable_jsonl(
        CONTROL_ROOT / "GOAL_EVENTS.jsonl",
        CONTROL_ROOT / "event_records" / f"000000_{txn['event']['event_sha256']}.jsonl",
        txn["event"],
        lease=lease,
    )
    # GOAL_STATE is the last commit marker. A crash before it is recoverable from intent/plan/event.
    atomic_write_bytes(
        CONTROL_ROOT / "CONTINUATION.md",
        txn["continuation"],
        lease=lease,
        precondition="ABSENT_OR_IDENTICAL",
    )
    atomic_write_bytes(
        CONTROL_ROOT / "GOAL_STATE.json",
        txn["state_bytes"],
        lease=lease,
        precondition="ABSENT_OR_IDENTICAL",
    )


def _event_from_line(encoded: bytes) -> dict[str, Any]:
    event = json.loads(encoded.decode("utf-8"))
    if canonical_json_bytes(event) != encoded:
        raise PreflightError("event journal contains non-canonical JSON")
    if type(event.get("seq")) is not int:
        raise PreflightError("event seq is not an exact integer")
    if event.get("event_sha256") != semantic_sha256(event, excluded_fields=("event_sha256",)):
        raise PreflightError("event semantic hash mismatch")
    return event


def verify_g0_state(bootstrap_lease: BootstrapLease) -> dict[str, Any]:
    bootstrap_lease.assert_active_capability()
    validate_bootstrap_control_parent()
    preverify_protected_metadata = collect_protected_metadata_records()
    _assert_control_namespace_not_protected(preverify_protected_metadata)
    required = {
        "g0_bootstrap_intent.json",
        "g0_bootstrap_intent.root.sha256",
        "execution_scope.json",
        "authority_ledger.json",
        "evidence_lock.json",
        "baseline_registry.json",
        "gate_registry.json",
        "budget.json",
        "rollback_map.json",
        "preexisting_dirty_inventory.json",
        "artifact_manifest.json",
        "artifact_manifest.root.sha256",
        "GOAL_EVENTS.jsonl",
        "GOAL_STATE.json",
        "CONTINUATION.md",
        "single_writer.lock",
        "single_writer.heartbeat.json",
    }
    pre_namespace = validate_namespace_ownership(allow_partial_control=True)
    existing = {name for name in pre_namespace["control_files"] if "/" not in name}
    missing = sorted(required - existing)
    if missing:
        raise PreflightError(f"missing mandatory G0 control artifacts: {missing}")
    intent = _load_and_verify_intent(
        CONTROL_ROOT / "g0_bootstrap_intent.json",
        CONTROL_ROOT / "g0_bootstrap_intent.root.sha256",
    )
    txn = _prepare_transaction(intent)
    namespace = validate_namespace_ownership(allow_partial_control=True)
    if namespace["recovery_temps"]:
        raise PreflightError(f"unresolved recovery temp files: {namespace['recovery_temps']}")
    expected_files, expected_directories = _g0_expected_namespace(txn, intent)
    namespace_entries = _walk_namespace(CONTROL_ROOT)
    actual_files = {
        path.relative_to(CONTROL_ROOT).as_posix()
        for path, object_type in namespace_entries
        if object_type == "file"
    }
    actual_directories = {
        path.relative_to(CONTROL_ROOT).as_posix()
        for path, object_type in namespace_entries
        if object_type == "directory"
    }
    if actual_files != expected_files or actual_directories != expected_directories:
        raise PreflightError(
            "G0 control namespace is not the exact intent/event reference closure: "
            f"extra_files={sorted(actual_files - expected_files)} "
            f"missing_files={sorted(expected_files - actual_files)} "
            f"extra_dirs={sorted(actual_directories - expected_directories)} "
            f"missing_dirs={sorted(expected_directories - actual_directories)}"
        )
    lock_bytes, lock_record = read_regular_bytes(
        CONTROL_ROOT / "single_writer.lock",
        max_bytes=128,
        require_nlink_one=True,
    )
    if lock_bytes != b"C28F_SINGLE_WRITER_LOCK_V2\n" or (
        int(lock_record["device"]), int(lock_record["inode"])
    ) != (
        intent["writer_lease"]["device"],
        intent["writer_lease"]["inode"],
    ) or lock_record["mode"] != intent["writer_lease"]["mode"] or lock_record["nlink"] != 1:
        raise PreflightError("single-writer fixed lock does not match the G0 intent binding")
    heartbeat = load_json(CONTROL_ROOT / "single_writer.heartbeat.json")
    required_heartbeat_keys = {
        "schema_version",
        "goal_id",
        "lease_id",
        "lease_lineage_root_id",
        "status",
        "pid",
        "boot_id",
        "process_start_ticks",
        "action",
        "acquired_at_ns",
        "heartbeat_at_ns",
    }
    allowed_heartbeat_keys = required_heartbeat_keys | {
        "hostname",
        "previous_owner",
        "phase",
        "last_action_result",
        "released_at_ns",
    }
    if (
        not isinstance(heartbeat, dict)
        or not required_heartbeat_keys.issubset(heartbeat)
        or not set(heartbeat).issubset(allowed_heartbeat_keys)
        or heartbeat["schema_version"] != "c28f_single_writer_lease_v2"
        or heartbeat["goal_id"] != GOAL_ID
        or heartbeat["lease_lineage_root_id"] != intent["writer_lease"]["lease_lineage_root_id"]
        or not isinstance(heartbeat["lease_id"], str)
        or re.fullmatch(r"[0-9a-f]{32}", heartbeat["lease_id"]) is None
        or heartbeat["status"] not in {"ACTIVE", "IDLE"}
        or heartbeat["action"] != "G0_SCOPE_LOCK_ATTEMPT_3"
        or type(heartbeat["pid"]) is not int
        or type(heartbeat["process_start_ticks"]) is not int
    ):
        raise PreflightError("single-writer heartbeat schema/lineage is invalid")
    if heartbeat["status"] == "ACTIVE":
        current_start_ticks = int(
            _proc_read(Path(f"/proc/{os.getpid()}/stat"), 8192)
            .decode("utf-8", "replace")
            .split()[21]
        )
        if heartbeat["pid"] != os.getpid() or heartbeat["process_start_ticks"] != current_start_ticks:
            raise PreflightError("ACTIVE single-writer heartbeat is not owned by this verifier")
    plan_path = CONTROL_ROOT / "transactions" / intent["transaction_id"] / "plan.json"
    plan_bytes, _plan_record = read_regular_bytes(plan_path, require_nlink_one=True)
    if plan_bytes != txn["plan_bytes"]:
        raise PreflightError("G0 transaction plan mismatch")
    state = load_json(CONTROL_ROOT / "GOAL_STATE.json")
    if state != txn["state"] or state.get("state_sha256") != semantic_sha256(
        state, excluded_fields=("state_sha256",)
    ):
        raise PreflightError("GOAL_STATE mismatch")
    state_record, _record = read_regular_bytes(
        CONTROL_ROOT / "state_records" / f"000000_{state['state_sha256']}.json",
        require_nlink_one=True,
    )
    if state_record != txn["state_bytes"]:
        raise PreflightError("immutable state record mismatch")
    journal, _journal_record = read_regular_bytes(
        CONTROL_ROOT / "GOAL_EVENTS.jsonl", require_nlink_one=True
    )
    if not journal.endswith(b"\n") or b"\n\n" in journal:
        raise PreflightError("event journal framing is invalid")
    lines = journal[:-1].split(b"\n")
    if len(lines) != 1:
        raise PreflightError(f"G0 verifier expected one committed event, found {len(lines)}")
    event = _event_from_line(lines[0])
    if event != txn["event"] or event["prev_event_sha256"] is not None:
        raise PreflightError("GENESIS event mismatch")
    sidecar, _sidecar_record = read_regular_bytes(
        CONTROL_ROOT / "event_records" / f"000000_{event['event_sha256']}.jsonl",
        require_nlink_one=True,
    )
    if sidecar != txn["event_bytes"]:
        raise PreflightError("GENESIS event sidecar mismatch")
    for name, expected_sha in event["output_hashes"].items():
        data, record = read_regular_bytes(CONTROL_ROOT / name, require_nlink_one=True)
        if name in txn["registry_bytes"]:
            expected_data = txn["registry_bytes"][name]
        elif name == "artifact_manifest.json":
            expected_data = txn["artifact_bytes"]
        elif name == "artifact_manifest.root.sha256":
            expected_data = txn["artifact_root_bytes"]
        else:
            raise PreflightError(f"GENESIS event declares unknown output: {name}")
        if record["sha256"] != expected_sha or data != expected_data:
            raise PreflightError(f"GENESIS output mismatch: {name}")
    for name, expected_data in txn["registry_bytes"].items():
        digest = bytes_sha256(expected_data)
        safe_name = name.replace(".json", "")
        version, _version_record = read_regular_bytes(
            CONTROL_ROOT / "registry_records" / f"000000_{safe_name}_{digest}.json",
            require_nlink_one=True,
        )
        if version != expected_data:
            raise PreflightError(f"immutable registry version mismatch: {name}")
    artifact_digest = bytes_sha256(txn["artifact_bytes"])
    artifact_version, _artifact_version_record = read_regular_bytes(
        CONTROL_ROOT
        / "registry_records"
        / f"000000_artifact_manifest_{artifact_digest}.json",
        require_nlink_one=True,
    )
    if artifact_version != txn["artifact_bytes"]:
        raise PreflightError("immutable artifact manifest version mismatch")
    continuation, _record = read_regular_bytes(
        CONTROL_ROOT / "CONTINUATION.md", require_nlink_one=True
    )
    if continuation != txn["continuation"]:
        raise PreflightError("CONTINUATION is not derived from current state/event")
    artifact, artifact_record = read_regular_bytes(
        CONTROL_ROOT / "artifact_manifest.json", require_nlink_one=True
    )
    root, _root_record = read_regular_bytes(
        CONTROL_ROOT / "artifact_manifest.root.sha256",
        max_bytes=256,
        require_nlink_one=True,
    )
    if root != (artifact_record["sha256"] + "\n").encode("ascii"):
        raise PreflightError("artifact manifest detached root mismatch")
    protected_metadata_now = collect_protected_metadata_records()
    protected_identities_now = _identity_pairs(protected_metadata_now)
    current_business = business_inventory(forbidden_identities=protected_identities_now)
    if current_business != intent["bootstrap_business_inventory"]:
        raise PreflightError("business delta drift after GENESIS")
    current_git = collect_git_inventory(protected_metadata_now)
    if current_git["branch"] != EXPECTED_BRANCH or current_git["head"] != EXPECTED_HEAD:
        raise PreflightError("branch/HEAD drift")
    if current_git["preexisting_dirty_inventory_sha256"] != intent["git_baseline"][
        "preexisting_dirty_inventory_sha256"
    ]:
        raise PreflightError("unexpected external dirty-state delta")
    authority_now = collect_authority_files(protected_metadata_now)
    immutable_now = collect_immutable_lock(authority_now, protected_metadata_now)
    if immutable_now != intent["immutable_lock"]:
        raise PreflightError("immutable input lock drift")
    process_now = collect_process_snapshot()
    if process_now["writer_conflicts"]:
        raise PreflightError("C28 writer conflict during G0 verification")
    authority = load_json(CONTROL_ROOT / "authority_ledger.json")
    if authority != _build_authority_ledger(intent):
        raise PreflightError("authority ledger payload mismatch")
    budget = load_json(CONTROL_ROOT / "budget.json")
    if budget != _build_budget(intent):
        raise PreflightError("budget/token payload mismatch")
    bootstrap_lease.assert_active_capability()
    return {
        "status": state["status"],
        "next_action": state["next_action"],
        "state_sha256": state["state_sha256"],
        "event_sha256": event["event_sha256"],
        "business_delta_sha256": current_business["business_delta_sha256"],
        "incident_disposition": "ACKNOWLEDGED_CONTINUE_WITH_RECORDED_EXCEPTION",
    }


def initialize_g0(
    producer_argv: list[str],
    producer_raw_argv: list[str],
    bootstrap_lease: BootstrapLease,
) -> dict[str, Any]:
    bootstrap_lease.assert_active_capability()
    if not PRE_G0_SECURITY_INCIDENT_USER_ACK:
        raise PreflightError("SECURITY_INCIDENT_AWAITING_USER_ACK")
    if (
        len(producer_argv) != 5
        or producer_argv[1] != "-B"
        or producer_argv[-2:] != ["--action", "G0_INIT"]
    ):
        raise PreflightError("G0 producer argv is not the frozen canonical invocation")
    if any(not Path(item).is_absolute() for item in (producer_argv[0], producer_argv[2])):
        raise PreflightError("Python executable and entrypoint must be absolute")
    if Path(producer_argv[2]) != ENTRYPOINT or Path(producer_argv[0]).resolve(strict=True) != Path(
        producer_argv[0]
    ):
        raise PreflightError("G0 producer executable/entrypoint paths are not canonical")
    if (
        len(producer_raw_argv) != 5
        or producer_raw_argv[1] != "-B"
        or producer_raw_argv[-2:] != ["--action", "G0_INIT"]
        or Path(producer_raw_argv[0]).resolve(strict=True) != Path(producer_argv[0])
        or Path(producer_raw_argv[2]).resolve(strict=True) != ENTRYPOINT
    ):
        raise PreflightError("G0 raw interpreter argv is not the real canonical -B invocation")
    validate_bootstrap_control_parent()
    prelease_protected_metadata = collect_protected_metadata_records()
    _assert_control_namespace_not_protected(prelease_protected_metadata)
    prelease_quarantine = collect_quarantined_g0_attempts(prelease_protected_metadata)
    if set(prelease_quarantine) != set(QUARANTINED_G0_ATTEMPTS):
        raise PreflightError("G0_INIT requires the exact committed quarantine receipt closure")
    validate_namespace_ownership(allow_partial_control=True)
    lock_path = CONTROL_ROOT / "single_writer.lock"
    heartbeat_path = CONTROL_ROOT / "single_writer.heartbeat.json"
    with WriterLease(lock_path, heartbeat_path, GOAL_ID, "G0_SCOPE_LOCK_ATTEMPT_3") as lease:
        cleanup_recovery_temps(lease=lease)
        lease.heartbeat("verify_authority_before_data")
        protected_metadata_records = collect_protected_metadata_records()
        protected_identities = _identity_pairs(protected_metadata_records)
        authority_files = collect_authority_files(protected_metadata_records)
        intent_path = CONTROL_ROOT / "g0_bootstrap_intent.json"
        intent_root = CONTROL_ROOT / "g0_bootstrap_intent.root.sha256"
        if intent_path.exists():
            intent = _load_and_verify_intent(
                intent_path,
                intent_root,
                require_detached_root=False,
            )
            if lease.fixed_lock_identity() != intent["writer_lease"]:
                raise PreflightError("current fixed lock/lease lineage does not match recovery intent")
            _write_intent(intent, lease=lease)
        else:
            validate_namespace_ownership(allow_partial_control=False)
            git = collect_git_inventory(protected_metadata_records)
            if git["branch"] != EXPECTED_BRANCH or git["head"] != EXPECTED_HEAD:
                raise PreflightError(f"branch/HEAD mismatch: {git['branch']} / {git['head']}")
            process = collect_process_snapshot()
            if process["writer_conflicts"]:
                raise PreflightError(f"C28 writer conflict: {process['writer_conflicts']}")
            immutable = collect_immutable_lock(authority_files, protected_metadata_records)
            business = business_inventory(forbidden_identities=protected_identities)
            _assert_quarantine_business_closure(
                business, immutable["quarantined_g0_attempts"]
            )
            incident = _incident_bundle()
            disk = shutil.disk_usage(REPO_ROOT)
            intent_base = {
                "schema_version": "c28f_g0_bootstrap_intent_v3",
                "goal_id": GOAL_ID,
                "goal_objective": GOAL_OBJECTIVE,
                "attempt_id": f"G0-A3-{uuid.uuid4().hex}",
                "transaction_id": f"G0-TXN-{uuid.uuid4().hex}",
                "created_at": _utc_now(),
                "created_at_ns": time.time_ns(),
                "producer_argv": producer_argv,
                "producer_raw_argv": producer_raw_argv,
                "producer_argv_sha256": bytes_sha256(canonical_json_bytes(producer_argv)),
                "producer_raw_argv_sha256": bytes_sha256(canonical_json_bytes(producer_raw_argv)),
                "writer_lease": lease.fixed_lock_identity(),
                "incident_bundle": incident,
                "immutable_lock": immutable,
                "git_baseline": git,
                "bootstrap_business_inventory": business,
                "process_snapshot": process,
                "disk": {
                    "total_bytes": int(disk.total),
                    "used_bytes": int(disk.used),
                    "free_bytes": int(disk.free),
                    "large_artifact_atomic_rule": "free >= 2*estimate + max(1GiB,10%)",
                },
            }
            intent = {
                **intent_base,
                "intent_sha256": bytes_sha256(canonical_json_bytes(intent_base)),
            }
            _validate_intent_contract(intent)
            preintent_txn = _prepare_transaction(intent)
            _assert_transaction_canonical_roundtrip(intent, preintent_txn)
            _write_intent(intent, lease=lease)
        if (
            intent["goal_id"] != GOAL_ID
            or intent["producer_argv"] != producer_argv
            or intent["producer_raw_argv"] != producer_raw_argv
        ):
            raise PreflightError("G0 intent goal/producer mismatch")
        lease.heartbeat("revalidate_all_inputs_before_genesis")
        validate_bootstrap_control_parent()
        if lease.fixed_lock_identity() != intent["writer_lease"]:
            raise PreflightError("writer lease binding drift before GENESIS")
        protected_metadata_now = collect_protected_metadata_records()
        protected_identities_now = _identity_pairs(protected_metadata_now)
        if collect_authority_files(protected_metadata_now) != intent["immutable_lock"]["authority_files"]:
            raise PreflightError("authority drift after intent")
        immutable_now = collect_immutable_lock(
            intent["immutable_lock"]["authority_files"],
            protected_metadata_now,
        )
        if immutable_now != intent["immutable_lock"]:
            raise PreflightError("immutable lock drift after intent")
        git_now = collect_git_inventory(protected_metadata_now)
        if (
            git_now["branch"] != EXPECTED_BRANCH
            or git_now["head"] != EXPECTED_HEAD
            or git_now["branch"] != intent["git_baseline"]["branch"]
            or git_now["head"] != intent["git_baseline"]["head"]
        ):
            raise PreflightError("branch/HEAD drift after G0 intent")
        if git_now["preexisting_dirty_inventory_sha256"] != intent["git_baseline"][
            "preexisting_dirty_inventory_sha256"
        ]:
            raise PreflightError("external dirty-state drift after intent")
        if business_inventory(forbidden_identities=protected_identities_now) != intent[
            "bootstrap_business_inventory"
        ]:
            raise PreflightError("bootstrap business delta drift after intent")
        process_now = collect_process_snapshot()
        if process_now["writer_conflicts"]:
            raise PreflightError("C28 writer conflict before GENESIS")
        txn = _prepare_transaction(intent)
        _assert_transaction_canonical_roundtrip(intent, txn)
        _assert_g0_recovery_namespace_subset(txn, intent)
        lease.heartbeat("install_g0_transaction")
        validate_bootstrap_control_parent()
        if lease.fixed_lock_identity() != intent["writer_lease"]:
            raise PreflightError("writer lease binding drift immediately before transaction install")
        _install_transaction(
            txn,
            intent,
            lease=lease,
            protected_identities=protected_identities_now,
        )
        lease.heartbeat("verify_g0_transaction")
        return verify_g0_state(bootstrap_lease)
