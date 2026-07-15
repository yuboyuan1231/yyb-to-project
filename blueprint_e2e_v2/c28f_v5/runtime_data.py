"""Reviewed data projections and one-pass completed-forward derivation.

This module has no state machine.  It validates the imported TRAIN index,
performs exact indexed ``pread`` projections, and turns sealed forward chunks
into metric-sufficient rows while retaining at most one raw chunk at a time.
It deliberately never imports the superseded :mod:`a4_control` module.
"""

from __future__ import annotations

import json
import math
import os
import stat
import string
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from .canonical import (
    bytes_sha256,
    canonical_json_bytes,
    read_regular_bytes,
    semantic_sha256,
)
from .constants import (
    CALIB_HOLDOUT_MANIFEST,
    CONTROL_ROOT,
    GOAL_ID,
    PSEUDO_OFFICIAL_MANIFEST,
    TRAIN_FIT_MANIFEST,
    TRAIN_JSONL,
)
from .runtime_store import RuntimeControlError, RuntimeStore


TRAIN_INDEX_TRANSACTION = "TRAIN-INDEX-b4e5edd34fa82b416b8e93ba31d40ffc"
FROZEN_SUPERSESSION_BRIDGE_SHA256 = (
    "ea22ea92671a2c5aa05c92f4f19db2efa8ad013f0d80249f629e984a867afbac"
)
PROTECTED_MANIFEST_SPECS = (
    (
        CALIB_HOLDOUT_MANIFEST,
        "7d5dbe30a18ef6e6320a63521e872c362689a163c937af1d43038e73d34c2afa",
        4_340,
        "155bf31f575eaa4f9f161c34830b9020da03bdd8e63de68f0d8efae09c8a9802",
    ),
    (
        PSEUDO_OFFICIAL_MANIFEST,
        "d250205897b827c964d0a59326dc2d63163c528d983aec1c3f955823fd17fc14",
        4_339,
        "7495f810cf0385e1be963c5c9e2883327dc946ca681dbf7260b6ce565e441340",
    ),
)
PROTECTED_UNION_COUNT = 8_679
PROTECTED_UNION_SHA256 = (
    "39a758494985a4db0963d5bcf0da92738b57f04b913a147b328532edceb2db31"
)
TRAIN_FIT_UNIVERSE_COUNT = 69_428
TRAIN_FIT_MANIFEST_FILE_SHA256 = (
    "24007d3f6f1159f55e68a8ee64c779436a34db4bff601f9d9cd74ee6e102bf0b"
)

QUERY_BUCKET_RULE_CONTRACT_BASE = {
    "schema_version": "c28f_a4_eval_bucket_rule_contract_v1",
    "contract_version": "FROZEN_PRE_EVAL_V1",
    "query_feature_token_count_rule": "FORWARD_SEALED_QUERY_MASK_SUM_USED_BY_MODEL",
    "late_maxsim_token_count_rule": "FORWARD_SEALED_LATE_MAXSIM_QUERY_MASK_SUM",
    "feature_to_late_maxsim_invariant": "EXACT_SAME_FORWARD_QUERY_MASK_COUNTS_EQUAL",
    "query_text_token_count_rule": (
        "UNICODE_CASEFOLD_THEN_ASCII_PUNCTUATION_TO_SPACE_THEN_"
        "WHITESPACE_SPLIT_COUNT"
    ),
    "verb_bucket_rule": "FROZEN_LEXICON_MATCH_COUNT_0_1_GE2",
    "verb_lexicon": [
        "be", "begin", "bring", "call", "carry", "come", "cut", "do",
        "drive", "eat", "enter", "fall", "find", "get", "give", "go",
        "grab", "hold", "jump", "keep", "leave", "look", "make", "move",
        "open", "pick", "play", "pull", "put", "read", "remove", "return",
        "run", "say", "see", "show", "sit", "stand", "start", "stop",
        "take", "talk", "throw", "turn", "use", "walk", "watch", "wear",
    ],
    "temporal_connector_rule": "ANY_EXACT_CASEFOLDED_TOKEN_MATCH",
    "temporal_connector_tokens": [
        "after", "before", "during", "finally", "first", "later", "next",
        "then", "until", "while",
    ],
    "duration_bucket_edges_sec": [2.0, 5.0, 10.0],
    "gt_video_clip_count_rule": "MAX_1_CEIL_GT_VIDEO_DURATION_SEC_DIV_1P5",
    "clip_count_bucket_edges": [64, 96],
    "relative_position_bucket_edges": [1.0 / 3.0, 2.0 / 3.0],
    "mask_coverage_bucket_edges": [0.25, 0.5, 0.75, 1.0],
    "margin_null_rule": "CUT_MARGIN_NONNULL;GT_MARGIN_NULL_IFF_GT_ABSENT",
    "teacher_rule": "UNAVAILABLE_ZERO_TEACHER_ACCESS",
}
QUERY_BUCKET_RULE_CONTRACT = {
    **QUERY_BUCKET_RULE_CONTRACT_BASE,
    "contract_sha256": semantic_sha256(QUERY_BUCKET_RULE_CONTRACT_BASE),
}


def _canonical_file(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _read_json(path: Path, *, max_bytes: int = 1024 * 1024 * 1024) -> dict[str, Any]:
    data, _record = read_regular_bytes(
        path, max_bytes=max_bytes, require_nlink_one=True
    )
    try:
        value = json.loads(data.decode("utf-8", "strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeControlError(f"invalid canonical JSON evidence: {path}") from exc
    if not isinstance(value, dict) or data != _canonical_file(value):
        raise RuntimeControlError(f"non-canonical JSON evidence: {path}")
    return value


def _fd_mount_id(fd: int) -> int:
    data = Path(f"/proc/self/fdinfo/{fd}").read_bytes()
    matches = [
        line.split(b":", 1)[1].strip()
        for line in data.splitlines()
        if line.startswith(b"mnt_id:")
    ]
    if len(matches) != 1 or not matches[0].isdigit():
        raise RuntimeControlError("file descriptor mount identity unavailable")
    return int(matches[0])


def _frame_indexed_jsonl_record(
    encoded: bytes,
    *,
    offset: int,
    length: int,
    source_size_bytes: int,
) -> bytes:
    """Supply the pure projector's delimiter for the indexed EOF record.

    The committed TRAIN index explicitly accepts one unterminated final JSONL
    record.  Only that descriptor-bound EOF record may receive a virtual
    delimiter; physical byte accounting continues to use ``encoded``.
    """

    if (
        type(encoded) is not bytes
        or type(offset) is not int
        or offset < 0
        or type(length) is not int
        or length < 1
        or type(source_size_bytes) is not int
        or source_size_bytes < 1
        or len(encoded) != length
        or offset + length > source_size_bytes
    ):
        raise RuntimeControlError("TRAIN indexed pread framing drift")
    if encoded.endswith(b"\n"):
        return encoded
    if (
        offset + length != source_size_bytes
        or len(encoded) + 1 > 1_048_576
    ):
        raise RuntimeControlError("TRAIN indexed pread framing drift")
    return encoded + b"\n"


class ImportedTrainIndex:
    """Exact read-only projection of the already committed TRAIN index."""

    def __init__(self, store: RuntimeStore | None = None) -> None:
        runtime = store or RuntimeStore()
        bridge = _read_json(
            runtime.root / "control_bridge" / "supersession_bridge.json"
        )
        if bridge.get("bridge_sha256") != semantic_sha256(
            bridge, ("bridge_sha256",)
        ) or bridge.get("bridge_sha256") != FROZEN_SUPERSESSION_BRIDGE_SHA256:
            raise RuntimeControlError("supersession bridge self-hash drift")
        gate = bridge.get("gate_projection", {}).get("TRAIN_JSONL_INDEX")
        if (
            not isinstance(gate, dict)
            or gate.get("transaction_id") != TRAIN_INDEX_TRANSACTION
            or gate.get("status") != "COMPLETED_NONREPLAYABLE_REPROJECTED"
            or gate.get("replay_allowed") is not False
        ):
            raise RuntimeControlError("imported TRAIN index gate is invalid")
        root = CONTROL_ROOT / "transactions" / TRAIN_INDEX_TRANSACTION
        index_path = root / "train_jsonl_index.json"
        index_data, _record = read_regular_bytes(
            index_path,
            max_bytes=1024 * 1024 * 1024,
            require_nlink_one=True,
        )
        try:
            index = json.loads(index_data.decode("utf-8", "strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeControlError("committed TRAIN index JSON is invalid") from exc
        receipt = _read_json(root / "bootstrap_receipt.json")
        if (
            not isinstance(index, dict)
            or index_data != _canonical_file(index)
            or bytes_sha256(index_data) != gate.get("index_file_sha256")
            or index.get("schema_version")
            != "c28f_a4_train_jsonl_descriptor_index_v1"
            or index.get("status") != "FSYNCED_DESCRIPTOR_BOUND_INDEX_POSTIMAGE"
            or index.get("goal_id") != GOAL_ID
            or index.get("transaction_id") != TRAIN_INDEX_TRANSACTION
            or index.get("index_binding_sha256")
            != semantic_sha256(index, ("index_binding_sha256",))
            or index.get("index_binding_sha256") != gate.get("index_binding_sha256")
            or index.get("entries_sha256")
            != bytes_sha256(canonical_json_bytes(index.get("entries")))
            or receipt.get("receipt_sha256")
            != semantic_sha256(receipt, ("receipt_sha256",))
            or receipt.get("receipt_sha256") != gate.get("receipt_sha256")
            or receipt.get("index_file_sha256") != bytes_sha256(index_data)
            or receipt.get("source_file_sha256") != index.get("source_file_sha256")
            or receipt.get("replay_allowed") is not False
        ):
            raise RuntimeControlError("committed TRAIN index binding drift")
        entries = index.get("entries")
        if not isinstance(entries, list) or len(entries) != index.get("line_count"):
            raise RuntimeControlError("committed TRAIN index entry count drift")
        by_desc: dict[int, dict[str, int]] = {}
        next_offset = 0
        ordered_entries = sorted(entries, key=lambda item: item.get("offset", -1))
        for entry in ordered_entries:
            if (
                not isinstance(entry, dict)
                or set(entry) != {"desc_id", "offset", "length"}
                or type(entry["desc_id"]) is not int
                or entry["desc_id"] < 0
                or type(entry["offset"]) is not int
                or entry["offset"] != next_offset
                or type(entry["length"]) is not int
                or not 1 < entry["length"] <= 1_048_576
                or entry["desc_id"] in by_desc
            ):
                raise RuntimeControlError("committed TRAIN index entry drift")
            by_desc[entry["desc_id"]] = entry
            next_offset += entry["length"]
        if (
            next_offset != index.get("source_size_bytes")
            or index.get("source_descriptor_identity_sha256")
            != semantic_sha256(index.get("source_descriptor", {}))
        ):
            raise RuntimeControlError("committed TRAIN index coverage drift")
        self.runtime = runtime
        self.index = index
        self.receipt = receipt
        self.by_desc = by_desc

    def project(
        self,
        desc_ids: Sequence[int],
        *,
        purpose: str,
        projection_scope: str,
    ) -> dict[str, Any]:
        from . import a4_id_projector

        retained = {
            "FORWARD_QUERY_IDENTITY": {"desc_id", "type", "desc_raw_json_sha256"},
            "POST_FORWARD_EVAL_STATS": {"desc_id", "desc", "type", "ts", "vid_name"},
            "TRAIN_FIT_ROLE_MAPPING": {"desc_id", "vid_name"},
            "PROTECTED_ID_MAPPING_ONLY": {"desc_id", "vid_name"},
        }
        requested = list(desc_ids)
        if (
            purpose not in retained
            or not projection_scope
            or Path(projection_scope).name != projection_scope
            or requested != sorted(set(requested))
            or any(type(value) is not int or value not in self.by_desc for value in requested)
        ):
            raise RuntimeControlError("authorized TRAIN projection request is invalid")
        seed = {
            "schema_version": "c28f_v5_runtime_train_projection_seed_v1",
            "goal_id": GOAL_ID,
            "purpose": purpose,
            "projection_scope": projection_scope,
            "requested_desc_ids_sha256": bytes_sha256(
                canonical_json_bytes(requested)
            ),
            "index_binding_sha256": self.index["index_binding_sha256"],
        }
        state = self.runtime.read_state()
        projection_id = "RUNTIME-PREAD-" + semantic_sha256(
            {**seed, "attempt_id": state["attempt_id"]}
        )[:32]

        descriptor = self.index["source_descriptor"]
        named = os.stat(TRAIN_JSONL, follow_symlinks=False)
        if (
            not stat.S_ISREG(named.st_mode)
            or int(named.st_dev) != descriptor["device"]
            or int(named.st_ino) != descriptor["inode"]
            or int(named.st_size) != descriptor["size_bytes"]
            or int(named.st_nlink) != descriptor["nlink"]
            or int(named.st_mtime_ns) != descriptor["mtime_ns"]
            or int(named.st_ctime_ns) != descriptor["ctime_ns"]
            or oct(stat.S_IMODE(named.st_mode)) != descriptor["mode"]
        ):
            raise RuntimeControlError("TRAIN source named identity drift")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(TRAIN_JSONL, flags)
        records: list[dict[str, Any]] = []
        total_bytes = 0
        try:
            before = os.fstat(fd)
            if (
                int(before.st_dev) != descriptor["device"]
                or int(before.st_ino) != descriptor["inode"]
                or int(before.st_size) != descriptor["size_bytes"]
                or _fd_mount_id(fd) != descriptor["mount_id"]
            ):
                raise RuntimeControlError("TRAIN source descriptor drift")
            for desc_id in requested:
                location = self.by_desc[desc_id]
                encoded = os.pread(fd, location["length"], location["offset"])
                projector_line = _frame_indexed_jsonl_record(
                    encoded,
                    offset=location["offset"],
                    length=location["length"],
                    source_size_bytes=descriptor["size_bytes"],
                )
                projected = a4_id_projector.extract_authorized_train_projection_line(
                    projector_line,
                    purpose=purpose,
                    expected_desc_id=desc_id,
                )
                record = projected.get("retained_record")
                counters = projected.get("counters")
                if (
                    not isinstance(record, dict)
                    or set(record) != retained[purpose]
                    or record.get("desc_id") != desc_id
                    or not isinstance(counters, dict)
                    or counters.get("full_line_json_decode_count") != 0
                    or counters.get("unauthorized_field_semantic_decode_count") != 0
                ):
                    raise RuntimeControlError("TRAIN value-minimizing parser drift")
                records.append(dict(record))
                total_bytes += len(encoded)
            after = os.fstat(fd)
            if (
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or _fd_mount_id(fd) != descriptor["mount_id"]
            ):
                raise RuntimeControlError("TRAIN source changed during indexed pread")
        finally:
            os.close(fd)
        access_base = {
            "schema_version": "c28f_v5_runtime_indexed_pread_receipt_v1",
            "status": "ONE_PHYSICAL_OPEN_EXACT_INDEXED_PREAD",
            "projection_id": projection_id,
            "purpose": purpose,
            "requested_desc_id_count": len(requested),
            "requested_desc_ids_sha256": seed["requested_desc_ids_sha256"],
            "physical_content_open_count": 1,
            "pread_call_count": len(requested),
            "pread_byte_count": total_bytes,
            "full_line_json_decode_count": 0,
            "unauthorized_field_semantic_decode_count": 0,
        }
        access = {**access_base, "receipt_sha256": semantic_sha256(access_base)}
        record_base = {
            "schema_version": "c28f_v5_runtime_train_projection_v1",
            "status": "IN_MEMORY_AUTHORIZED_INDEXED_PREAD_PROJECTION",
            "goal_id": GOAL_ID,
            "attempt_id": state["attempt_id"],
            "projection_id": projection_id,
            "purpose": purpose,
            "projection_scope": projection_scope,
            "index_binding_sha256": self.index["index_binding_sha256"],
            "index_receipt_sha256": self.receipt["receipt_sha256"],
            "source_descriptor_identity_sha256": self.index[
                "source_descriptor_identity_sha256"
            ],
            "source_file_sha256": self.index["source_file_sha256"],
            "records": records,
            "records_sha256": bytes_sha256(canonical_json_bytes(records)),
            "access_receipt": access,
            "access_receipt_sha256": access["receipt_sha256"],
        }
        return {**record_base, "record_sha256": semantic_sha256(record_base)}


def read_g4_manifest_ids() -> dict[str, Any]:
    """Open the three authorized ID manifests exactly once each."""

    train_data, train_record = read_regular_bytes(
        TRAIN_FIT_MANIFEST,
        max_bytes=8 * 1024 * 1024,
        require_nlink_one=True,
    )
    train_fit_ids = _decode_id_manifest(train_data)
    if (
        train_record["sha256"] != TRAIN_FIT_MANIFEST_FILE_SHA256
        or len(train_fit_ids) != TRAIN_FIT_UNIVERSE_COUNT
    ):
        raise RuntimeControlError("frozen train_fit desc-ID universe drift")

    protected_sets: list[list[int]] = []
    manifest_records: list[dict[str, Any]] = []
    for path, expected_file_sha, expected_count, expected_ids_sha in PROTECTED_MANIFEST_SPECS:
        data, record = read_regular_bytes(
            path, max_bytes=8 * 1024 * 1024, require_nlink_one=True
        )
        ids = _decode_id_manifest(data)
        if (
            record["sha256"] != expected_file_sha
            or len(ids) != expected_count
            or bytes_sha256(canonical_json_bytes(ids)) != expected_ids_sha
        ):
            raise RuntimeControlError("protected desc-ID manifest drift")
        protected_sets.append(ids)
        manifest_records.append(
            {
                "path": str(path),
                "file_sha256": record["sha256"],
                "desc_id_count": len(ids),
                "desc_ids_sha256": bytes_sha256(canonical_json_bytes(ids)),
            }
        )
    protected_ids = sorted(set().union(*(set(values) for values in protected_sets)))
    if (
        len(protected_ids) != PROTECTED_UNION_COUNT
        or bytes_sha256(canonical_json_bytes(protected_ids))
        != PROTECTED_UNION_SHA256
    ):
        raise RuntimeControlError("protected desc-ID union drift")
    return {
        "protected_desc_ids": protected_ids,
        "protected_desc_ids_sha256": bytes_sha256(canonical_json_bytes(protected_ids)),
        "train_fit_desc_ids": train_fit_ids,
        "train_fit_desc_ids_sha256": bytes_sha256(canonical_json_bytes(train_fit_ids)),
        "train_fit_manifest_file_sha256": train_record["sha256"],
        "manifest_records": manifest_records
        + [
            {
                "path": str(TRAIN_FIT_MANIFEST),
                "file_sha256": train_record["sha256"],
                "desc_id_count": len(train_fit_ids),
                "desc_ids_sha256": bytes_sha256(
                    canonical_json_bytes(train_fit_ids)
                ),
            }
        ],
        "physical_manifest_content_open_count": 3,
    }


def _decode_id_manifest(data: bytes) -> list[int]:
    if not data or not data.endswith(b"\n"):
        raise RuntimeControlError("ID manifest framing drift")
    encoded_ids = data[:-1].split(b"\n")
    if any(
        not value
        or not value.isdigit()
        or (value != b"0" and value.startswith(b"0"))
        for value in encoded_ids
    ):
        raise RuntimeControlError("ID manifest canonical decimal drift")
    ids = [int(value) for value in encoded_ids]
    if ids != sorted(set(ids)):
        raise RuntimeControlError("ID manifest ordering or uniqueness drift")
    return ids


def _temporal_iou(first: tuple[float, float], second: tuple[float, float]) -> float:
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = (first[1] - first[0]) + (second[1] - second[0]) - intersection
    if union <= 0.0:
        raise RuntimeControlError("temporal IoU received an invalid span")
    return intersection / union


def _derive_frozen_nms(metric_record: Mapping[str, Any]) -> dict[str, Any]:
    raw = [
        {
            "video_id": proposal["video_id"],
            "start_sec": float(proposal["start_sec"]),
            "end_sec": float(proposal["end_sec"]),
            "score": float(proposal["score"]),
            "probability": float(proposal["probability"]),
            "source_ordinal": ordinal,
        }
        for ordinal, proposal in enumerate(metric_record["joint_proposals"])
    ]
    raw.sort(
        key=lambda proposal: (
            -proposal["score"],
            proposal["video_id"],
            proposal["start_sec"],
            proposal["end_sec"],
            proposal["source_ordinal"],
        )
    )
    duplicate_count = len(raw) - len(
        {
            (item["video_id"], item["start_sec"], item["end_sec"])
            for item in raw
        }
    )
    kept: list[dict[str, Any]] = []
    by_video: dict[str, list[dict[str, Any]]] = {}
    for proposal in raw:
        if any(
            _temporal_iou(
                (proposal["start_sec"], proposal["end_sec"]),
                (prior["start_sec"], prior["end_sec"]),
            )
            > 0.7
            for prior in by_video.get(proposal["video_id"], [])
        ):
            continue
        kept.append(dict(proposal))
        by_video.setdefault(proposal["video_id"], []).append(proposal)
        if len(kept) >= 100:
            break
    base = {
        "query_id": metric_record["query_id"],
        "predictions": kept,
        "prediction_count": len(kept),
        "raw_exact_duplicate_proposal_count": duplicate_count,
    }
    return {**base, "row_sha256": semantic_sha256(base)}


def _query_text_features(text: str) -> tuple[int, str, bool]:
    if not isinstance(text, str) or "\x00" in text:
        raise RuntimeControlError("completed eval query text is invalid")
    folded = text.casefold().translate(
        str.maketrans({character: " " for character in string.punctuation})
    )
    tokens = folded.split()
    if not tokens:
        raise RuntimeControlError("completed eval query text tokenization is empty")
    verbs = set(QUERY_BUCKET_RULE_CONTRACT["verb_lexicon"])
    temporal_tokens = set(QUERY_BUCKET_RULE_CONTRACT["temporal_connector_tokens"])
    verb_count = sum(token in verbs for token in tokens)
    bucket = "ZERO_VERB" if verb_count == 0 else "SINGLE_VERB" if verb_count == 1 else "MULTI_VERB"
    return len(tokens), bucket, any(token in temporal_tokens for token in tokens)


def _ranking_projection(
    ranking: Sequence[Mapping[str, Any]], *, gt_video_id: str
) -> tuple[list[str], dict[str, float | None]]:
    if len(ranking) < 101:
        raise RuntimeControlError("completed eval ranking lacks fixed cuts")
    order = [str(item["video_id"]) for item in ranking]
    scores = [float(item["score"]) for item in ranking]
    if (
        len(order) != len(set(order))
        or any(not math.isfinite(value) for value in scores)
        or any(scores[index] < scores[index + 1] for index in range(len(scores) - 1))
    ):
        raise RuntimeControlError("completed eval ranking order drift")
    margins: dict[str, float | None] = {}
    for label, first, second in (
        ("top1_top2", 0, 1),
        ("top10_top11", 9, 10),
        ("top100_top101", 99, 100),
    ):
        margin = scores[first] - scores[second]
        if not math.isfinite(margin) or margin < 0.0:
            raise RuntimeControlError("completed eval ranking margin drift")
        margins[label] = margin
    if gt_video_id not in order:
        margins["gt_vs_best_wrong"] = None
    else:
        gt_score = scores[order.index(gt_video_id)]
        best_wrong = next(score for video_id, score in zip(order, scores) if video_id != gt_video_id)
        margins["gt_vs_best_wrong"] = gt_score - best_wrong
    return order, margins


def _joint_stats(
    metric_record: Mapping[str, Any],
    nms_row: Mapping[str, Any],
    *,
    gt_video_id: str,
    gt_span: tuple[float, float],
    final_video_order: Sequence[str],
) -> dict[str, Any]:
    probability_mass: dict[str, float] = {}
    false_positive_parts: list[float] = []
    for proposal in metric_record["joint_proposals"]:
        video_id = str(proposal["video_id"])
        probability = float(proposal["probability"])
        probability_mass[video_id] = probability_mass.get(video_id, 0.0) + probability
        span = (float(proposal["start_sec"]), float(proposal["end_sec"]))
        if video_id != gt_video_id or _temporal_iou(span, gt_span) < 0.3:
            false_positive_parts.append(probability)
    if (
        not math.isclose(math.fsum(probability_mass.values()), 1.0, rel_tol=0.0, abs_tol=1e-8)
        or set(probability_mass) != set(final_video_order)
    ):
        raise RuntimeControlError("completed eval joint probability closure drift")
    marginal = sorted(probability_mass, key=lambda key: (-probability_mass[key], key))
    marginal_rank = marginal.index(gt_video_id) + 1 if gt_video_id in marginal else None
    predictions = list(nms_row["predictions"])
    unique_order: list[str] = []
    for prediction in predictions:
        video_id = str(prediction["video_id"])
        if video_id not in unique_order:
            unique_order.append(video_id)
    legacy_rank = unique_order.index(gt_video_id) + 1 if gt_video_id in unique_order else None
    hits = {
        f"VCMR_R@{k}_IoU@{threshold:.1f}": any(
            str(prediction["video_id"]) == gt_video_id
            and _temporal_iou(
                (float(prediction["start_sec"]), float(prediction["end_sec"])),
                gt_span,
            )
            >= threshold
            for prediction in predictions[:k]
        )
        for threshold in (0.5, 0.7)
        for k in (1, 5, 10, 100)
    }
    available = bool(predictions)
    wrong_video = not available or str(predictions[0]["video_id"]) != gt_video_id
    top1_iou = (
        _temporal_iou(
            (float(predictions[0]["start_sec"]), float(predictions[0]["end_sec"])),
            gt_span,
        )
        if available and not wrong_video
        else 0.0
    )
    false_positive_mass = math.fsum(false_positive_parts) if available else None
    if false_positive_mass is not None:
        if not -1e-12 <= false_positive_mass <= 1.0 + 1e-12:
            raise RuntimeControlError(
                "completed eval false-positive probability mass drift"
            )
        false_positive_mass = min(1.0, max(0.0, false_positive_mass))
    return {
        "joint_marginal_gt_rank": marginal_rank,
        "legacy_joint_unique_video_gt_rank": legacy_rank,
        "vcmr_hits": hits,
        "top1_iou": top1_iou,
        "top1_available": available,
        "wrong_video_top1": wrong_video,
        "correct_video_wrong_span_top1": available and not wrong_video and top1_iou < 0.5,
        "top1_false_positive": (not available) or wrong_video or top1_iou < 0.3,
        "joint_false_positive_mass": false_positive_mass,
        "raw_exact_duplicate_proposal_count": nms_row[
            "raw_exact_duplicate_proposal_count"
        ],
        "nms_prediction_count": nms_row["prediction_count"],
    }


def _retrieval_row(
    metric_record: Mapping[str, Any],
    nms_row: Mapping[str, Any],
    gt: Mapping[str, Any],
    *,
    duration_by_video: Mapping[str, float],
) -> dict[str, Any]:
    query_id = int(metric_record["query_id"])
    gt_video_id = str(gt["vid_name"])
    if gt.get("type") != metric_record["query_type"] or gt_video_id not in duration_by_video:
        raise RuntimeControlError("completed eval GT query/corpus binding drift")
    gt_start, gt_end = float(gt["ts"][0]), float(gt["ts"][1])
    duration = float(duration_by_video[gt_video_id])
    if not 0.0 <= gt_start < gt_end <= duration + 1e-6:
        raise RuntimeControlError("completed eval GT timestamp drift")
    text_count, verb_bucket, temporal = _query_text_features(str(gt["desc"]))
    ranking_values: dict[str, Any] = {}
    for prefix, field in (
        ("pooled", "pooled_video_order"),
        ("late", "late_video_order"),
        ("final", "final_video_order"),
    ):
        order, margins = _ranking_projection(
            metric_record[field], gt_video_id=gt_video_id
        )
        ranking_values[field] = order
        for name, value in margins.items():
            ranking_values[f"{prefix}_{name}_score_margin"] = value
    forensics = metric_record["forensics"]
    return {
        "query_id": query_id,
        "gt_video_id": gt_video_id,
        **ranking_values,
        "query_type": metric_record["query_type"],
        "query_feature_token_count": forensics["query_feature_token_count"],
        "query_text_token_count": text_count,
        "verb_bucket": verb_bucket,
        "temporal_connector_bool": temporal,
        "late_maxsim_token_count": forensics["late_maxsim_token_count"],
        "late_candidate_mask_coverage": forensics["late_candidate_mask_coverage"],
        "moment_duration_sec": gt_end - gt_start,
        "gt_video_clip_count": max(1, int(math.ceil(duration / 1.5))),
        "gt_relative_position": min(
            1.0, max(0.0, ((gt_start + gt_end) / 2.0) / duration)
        ),
        **_joint_stats(
            metric_record,
            nms_row,
            gt_video_id=gt_video_id,
            gt_span=(gt_start, gt_end),
            final_video_order=ranking_values["final_video_order"],
        ),
    }


def derive_completed_forward_rows(
    material_path: Path,
    gt_projection: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate and consume every sealed raw chunk exactly once."""

    from . import a4_forward

    material = _read_json(material_path)
    if material.get("material_binding_sha256") != semantic_sha256(
        material, ("material_binding_sha256",)
    ):
        raise RuntimeControlError("completed forward material binding drift")
    projection = material.get("completed_result_verification_projection")
    validation = material.get("completed_forward_validation_receipt")
    chunk_root_view = material.get("chunk_root")
    if not all(isinstance(value, dict) for value in (projection, validation, chunk_root_view)):
        raise RuntimeControlError("completed forward material is incomplete")
    authority = a4_forward._rehydrate_completed_forward_authority_from_projection(
        projection
    )
    manifest = authority.manifest
    if (
        validation.get("validation_receipt_sha256")
        != semantic_sha256(validation, ("validation_receipt_sha256",))
        or validation.get("query_count") != len(manifest.query_identity.query_keys)
        or validation.get("query_rows_sha256")
        != manifest.query_identity.query_rows_sha256
        or validation.get("query_types_sha256")
        != manifest.query_identity.query_types_sha256
        or validation.get("result_chunk_count")
        != len(validation.get("result_chunk_index", []))
        or validation.get("maximum_simultaneously_loaded_chunk_count") != 1
    ):
        raise RuntimeControlError("completed forward validation receipt drift")
    gt_records = gt_projection.get("records")
    if (
        not isinstance(gt_records, list)
        or gt_projection.get("purpose") != "POST_FORWARD_EVAL_STATS"
        or gt_projection.get("records_sha256")
        != bytes_sha256(canonical_json_bytes(gt_records))
    ):
        raise RuntimeControlError("completed forward GT projection drift")
    gt_by_id = {int(record["desc_id"]): record for record in gt_records}
    query_ids = [int(value) for value in manifest.query_identity.query_keys]
    if list(gt_by_id) != query_ids:
        raise RuntimeControlError("completed forward GT projection universe drift")
    duration_by_video = dict(
        zip(manifest.corpus_identity.video_ids, manifest.corpus_identity.durations_sec)
    )
    chunk_root = a4_forward.FrozenDirectory(**dict(chunk_root_view))
    capability = SimpleNamespace(
        capability_id=validation["capability_id"],
        token_id=validation["token_id"],
        eval_id=validation["eval_id"],
    )
    torch_module = a4_forward._lazy_torch()
    rows: list[dict[str, Any]] = []
    chunk_receipts: list[dict[str, Any]] = []
    expected_start = 0
    for expected_range_index, chunk in enumerate(validation["result_chunk_index"]):
        if (
            chunk.get("range_index") != expected_range_index
            or chunk.get("start_index") != expected_start
            or chunk.get("end_index") - chunk.get("start_index")
            != chunk.get("row_count")
        ):
            raise RuntimeControlError("completed forward chunk index drift")
        path = Path(chunk_root.path) / str(chunk["artifact_name"])
        payload = a4_forward._safe_load_owned_torch_artifact(
            path,
            torch_module,
            expected_parent=chunk_root,
            expected_file_sha256=chunk["artifact_file_sha256"],
        )
        if not isinstance(payload, Mapping):
            raise RuntimeControlError("completed forward chunk payload is invalid")
        a4_forward._validate_forward_result_payload(
            torch_module,
            payload,
            authority=authority,
            capability=capability,
            expected_start=chunk["start_index"],
            expected_end=chunk["end_index"],
        )
        if (
            a4_forward._forward_result_payload_digest(torch_module, payload)
            != chunk["payload_semantic_sha256"]
            or payload.get("engine_registry_consumption_sha256")
            != chunk.get("engine_registry_consumption_sha256")
        ):
            raise RuntimeControlError("completed forward chunk semantic digest drift")
        for raw_row in payload["rows"]:
            metric_record = a4_forward._controller_metric_record_from_forward_payload_row(
                raw_row,
                corpus=manifest.corpus_identity,
            )
            nms_row = _derive_frozen_nms(metric_record)
            query_id = int(metric_record["query_id"])
            rows.append(
                _retrieval_row(
                    metric_record,
                    nms_row,
                    gt_by_id[query_id],
                    duration_by_video=duration_by_video,
                )
            )
        chunk_receipts.append(
            {
                "range_index": expected_range_index,
                "artifact_name": chunk["artifact_name"],
                "artifact_file_sha256": chunk["artifact_file_sha256"],
                "payload_semantic_sha256": chunk["payload_semantic_sha256"],
                "start_index": chunk["start_index"],
                "end_index": chunk["end_index"],
            }
        )
        expected_start = chunk["end_index"]
        del payload
    if expected_start != len(query_ids) or [row["query_id"] for row in rows] != query_ids:
        raise RuntimeControlError("completed forward one-pass query coverage drift")
    lineage_base = {
        "schema_version": "c28f_v5_runtime_completed_forward_derivation_v1",
        "status": "ONE_PASS_ALL_SEALED_CHUNKS_VALIDATED",
        "token_id": validation["token_id"],
        "eval_id": validation["eval_id"],
        "run_id": validation["run_id"],
        "query_count": len(rows),
        "result_chunk_count": len(chunk_receipts),
        "result_chunk_set_sha256": validation["result_chunk_set_sha256"],
        "train_projection_sha256": gt_projection["record_sha256"],
        "train_projection_access_receipt_sha256": gt_projection[
            "access_receipt_sha256"
        ],
        "ordered_ranges_sha256": validation["ordered_ranges_sha256"],
        "chunk_receipts_sha256": bytes_sha256(
            canonical_json_bytes(chunk_receipts)
        ),
        "retrieval_rows_sha256": bytes_sha256(canonical_json_bytes(rows)),
        "maximum_simultaneously_loaded_chunk_count": 1,
        "raw_payload_retained_after_range": False,
        "nms_iou_operator": ">",
        "nms_iou_threshold": 0.7,
        "nms_max_keep": 100,
    }
    lineage = {**lineage_base, "lineage_sha256": semantic_sha256(lineage_base)}
    return rows, lineage


def canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_file(dict(row)) for row in rows)
