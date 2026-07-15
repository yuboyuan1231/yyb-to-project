"""Value-minimising TRAIN-JSONL projections with no source-opening authority.

This module owns no path or file opener.  Post-index legal-role access passes
one descriptor-bound ``pread`` line to a fixed-purpose pure parser.  A G4
protected-ID caller must additionally commit the independent
``AUTH_PROTECTED_ID_MAPPING_ONLY`` RUNNING operation and use the exact
controller-issued, single-use reader; that operation is not an F0-A/F0-B
evaluation and consumes no model-forward token.  Arbitrary source iterables
are rejected.  Non-target JSON values are syntax-scanned but never decoded,
materialised, logged, hashed separately, or returned.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from .a4_security import (
    MinimalIdMapRecord,
    PROTECTED_ID_MAPPING_RUNNING_ANCHOR_SCHEMA_VERSION,
    validate_protected_id_mapping_running_anchor,
)


PROJECTOR_SCHEMA_VERSION = "c28f_protected_mixed_json_id_projector_v1"
PROJECTION_MANIFEST_SCHEMA_VERSION = "c28f_protected_video_id_manifest_v1"
MAPPING_ACCESS_RECEIPT_SCHEMA_VERSION = "c28f_protected_id_mapping_access_receipt_v3"
PROJECTION_COMMIT_REQUEST_SCHEMA_VERSION = "c28f_g4_projection_commit_request_v1"
MAX_LINE_BYTES = 1_048_576
MAX_JSON_DEPTH = 64

_SAFE_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX_BYTES = frozenset(b"0123456789abcdefABCDEF")
_PROJECTION_RESULT_ISSUER = object()

AUTHORIZED_TRAIN_PROJECTION_PURPOSES = (
    "FORWARD_QUERY_IDENTITY",
    "POST_FORWARD_EVAL_STATS",
    "TRAIN_FIT_ROLE_MAPPING",
    "PROTECTED_ID_MAPPING_ONLY",
)

_TRAIN_PROJECTION_SOURCE_FIELDS = {
    "FORWARD_QUERY_IDENTITY": ("desc_id", "type", "desc"),
    "POST_FORWARD_EVAL_STATS": (
        "desc_id",
        "desc",
        "type",
        "ts",
        "vid_name",
    ),
    "TRAIN_FIT_ROLE_MAPPING": ("desc_id", "vid_name"),
    "PROTECTED_ID_MAPPING_ONLY": ("desc_id", "vid_name"),
}

_TRAIN_PROJECTION_RETAINED_FIELDS = {
    "FORWARD_QUERY_IDENTITY": (
        "desc_id",
        "type",
        "desc_raw_json_sha256",
    ),
    "POST_FORWARD_EVAL_STATS": (
        "desc_id",
        "desc",
        "type",
        "ts",
        "vid_name",
    ),
    "TRAIN_FIT_ROLE_MAPPING": ("desc_id", "vid_name"),
    "PROTECTED_ID_MAPPING_ONLY": ("desc_id", "vid_name"),
}


class IdProjectionError(RuntimeError):
    def __init__(self, code: str) -> None:
        if not isinstance(code, str) or re.fullmatch(r"[A-Z0-9_]+", code) is None:
            raise ValueError("projection error code must be an uppercase identifier")
        self.code = code
        super().__init__(code)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _semantic_sha256(value: Mapping[str, Any], excluded: Sequence[str] = ()) -> str:
    omitted = frozenset(excluded)
    return _sha256(
        _canonical_json_bytes({key: item for key, item in value.items() if key not in omitted})
    )


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise IdProjectionError("PROJECTION_LINEAGE_SHA_INVALID")
    return value


def _require_exact_mapping(
    value: Any,
    fields: Sequence[str],
    code: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise IdProjectionError(code)
    return value


def _skip_ws(data: bytes, offset: int) -> int:
    while offset < len(data) and data[offset] in b" \t\r\n":
        offset += 1
    return offset


def _expect(data: bytes, offset: int, expected: int, code: str) -> int:
    if offset >= len(data) or data[offset] != expected:
        raise IdProjectionError(code)
    return offset + 1


def _scan_plain_ascii_string(
    data: bytes, offset: int, *, max_bytes: int, code: str
) -> tuple[str, int]:
    offset = _expect(data, offset, ord('"'), code)
    start = offset
    while offset < len(data):
        current = data[offset]
        if current == ord('"'):
            raw = data[start:offset]
            if not raw or len(raw) > max_bytes:
                raise IdProjectionError(code)
            if any(value < 0x20 or value > 0x7E for value in raw) or b"\\" in raw:
                raise IdProjectionError(code)
            return raw.decode("ascii"), offset + 1
        if current == ord("\\") or current < 0x20 or current > 0x7E:
            raise IdProjectionError(code)
        offset += 1
    raise IdProjectionError(code)


def _scan_plain_ascii_token_bytes(
    data: bytes, offset: int, *, max_bytes: int, code: str
) -> tuple[bytes, int]:
    """Validate one plain-ASCII JSON string without semantic decoding.

    The post-index projector needs to route authorized top-level fields, but
    routing does not authorize materialising every source key as a Python
    string.  Returning the exact inner byte slice keeps non-selected key
    handling in the syntax/comparison domain.
    """

    offset = _expect(data, offset, ord('"'), code)
    start = offset
    while offset < len(data):
        current = data[offset]
        if current == ord('"'):
            raw = data[start:offset]
            if not raw or len(raw) > max_bytes:
                raise IdProjectionError(code)
            if any(value < 0x20 or value > 0x7E for value in raw) or b"\\" in raw:
                raise IdProjectionError(code)
            return raw, offset + 1
        if current == ord("\\") or current < 0x20 or current > 0x7E:
            raise IdProjectionError(code)
        offset += 1
    raise IdProjectionError(code)


def _consume_utf8_scalar(data: bytes, offset: int) -> int:
    first = data[offset]
    if 0xC2 <= first <= 0xDF:
        length = 2
        second_min, second_max = 0x80, 0xBF
    elif first == 0xE0:
        length = 3
        second_min, second_max = 0xA0, 0xBF
    elif 0xE1 <= first <= 0xEC or 0xEE <= first <= 0xEF:
        length = 3
        second_min, second_max = 0x80, 0xBF
    elif first == 0xED:
        length = 3
        second_min, second_max = 0x80, 0x9F
    elif first == 0xF0:
        length = 4
        second_min, second_max = 0x90, 0xBF
    elif 0xF1 <= first <= 0xF3:
        length = 4
        second_min, second_max = 0x80, 0xBF
    elif first == 0xF4:
        length = 4
        second_min, second_max = 0x80, 0x8F
    else:
        raise IdProjectionError("PROJECTOR_JSON_UTF8_INVALID")
    if offset + length > len(data):
        raise IdProjectionError("PROJECTOR_JSON_UTF8_TRUNCATED")
    if not second_min <= data[offset + 1] <= second_max:
        raise IdProjectionError("PROJECTOR_JSON_UTF8_INVALID")
    if any(not 0x80 <= value <= 0xBF for value in data[offset + 2 : offset + length]):
        raise IdProjectionError("PROJECTOR_JSON_UTF8_INVALID")
    return offset + length


def _escaped_utf16_code_unit(data: bytes, offset: int) -> tuple[int, int]:
    if offset + 4 > len(data) or any(value not in _HEX_BYTES for value in data[offset : offset + 4]):
        raise IdProjectionError("PROJECTOR_JSON_UNICODE_ESCAPE_INVALID")
    return int(data[offset : offset + 4].decode("ascii"), 16), offset + 4


def _skip_json_string(data: bytes, offset: int) -> int:
    offset = _expect(data, offset, ord('"'), "PROJECTOR_JSON_STRING_REQUIRED")
    while offset < len(data):
        current = data[offset]
        if current == ord('"'):
            return offset + 1
        if current < 0x20:
            raise IdProjectionError("PROJECTOR_JSON_CONTROL_CHARACTER")
        if current >= 0x80:
            offset = _consume_utf8_scalar(data, offset)
            continue
        if current == ord("\\"):
            offset += 1
            if offset >= len(data):
                raise IdProjectionError("PROJECTOR_JSON_ESCAPE_TRUNCATED")
            escaped = data[offset]
            if escaped == ord("u"):
                code_unit, offset = _escaped_utf16_code_unit(data, offset + 1)
                if 0xD800 <= code_unit <= 0xDBFF:
                    if data[offset : offset + 2] != b"\\u":
                        raise IdProjectionError("PROJECTOR_JSON_SURROGATE_PAIR_INVALID")
                    low, offset = _escaped_utf16_code_unit(data, offset + 2)
                    if not 0xDC00 <= low <= 0xDFFF:
                        raise IdProjectionError("PROJECTOR_JSON_SURROGATE_PAIR_INVALID")
                elif 0xDC00 <= code_unit <= 0xDFFF:
                    raise IdProjectionError("PROJECTOR_JSON_SURROGATE_PAIR_INVALID")
                continue
            if escaped not in b'"\\/bfnrt':
                raise IdProjectionError("PROJECTOR_JSON_ESCAPE_INVALID")
        offset += 1
    raise IdProjectionError("PROJECTOR_JSON_STRING_UNTERMINATED")


def _skip_json_number(data: bytes, offset: int) -> int:
    start = offset
    if offset < len(data) and data[offset] == ord("-"):
        offset += 1
    if offset >= len(data):
        raise IdProjectionError("PROJECTOR_JSON_NUMBER_INVALID")
    if data[offset] == ord("0"):
        offset += 1
        if offset < len(data) and ord("0") <= data[offset] <= ord("9"):
            raise IdProjectionError("PROJECTOR_JSON_NUMBER_INVALID")
    elif ord("1") <= data[offset] <= ord("9"):
        while offset < len(data) and ord("0") <= data[offset] <= ord("9"):
            offset += 1
    else:
        raise IdProjectionError("PROJECTOR_JSON_NUMBER_INVALID")
    if offset < len(data) and data[offset] == ord("."):
        offset += 1
        fraction_start = offset
        while offset < len(data) and ord("0") <= data[offset] <= ord("9"):
            offset += 1
        if offset == fraction_start:
            raise IdProjectionError("PROJECTOR_JSON_NUMBER_INVALID")
    if offset < len(data) and data[offset] in b"eE":
        offset += 1
        if offset < len(data) and data[offset] in b"+-":
            offset += 1
        exponent_start = offset
        while offset < len(data) and ord("0") <= data[offset] <= ord("9"):
            offset += 1
        if offset == exponent_start:
            raise IdProjectionError("PROJECTOR_JSON_NUMBER_INVALID")
    if offset == start:
        raise IdProjectionError("PROJECTOR_JSON_NUMBER_INVALID")
    return offset


def _skip_json_value(data: bytes, offset: int, *, depth: int) -> int:
    if depth > MAX_JSON_DEPTH:
        raise IdProjectionError("PROJECTOR_JSON_DEPTH_EXCEEDED")
    offset = _skip_ws(data, offset)
    if offset >= len(data):
        raise IdProjectionError("PROJECTOR_JSON_VALUE_MISSING")
    current = data[offset]
    if current == ord('"'):
        return _skip_json_string(data, offset)
    if current == ord("{"):
        offset += 1
        offset = _skip_ws(data, offset)
        if offset < len(data) and data[offset] == ord("}"):
            return offset + 1
        while True:
            offset = _skip_json_string(data, offset)
            offset = _skip_ws(data, offset)
            offset = _expect(data, offset, ord(":"), "PROJECTOR_JSON_COLON_REQUIRED")
            offset = _skip_json_value(data, offset, depth=depth + 1)
            offset = _skip_ws(data, offset)
            if offset < len(data) and data[offset] == ord(","):
                offset = _skip_ws(data, offset + 1)
                continue
            return _expect(data, offset, ord("}"), "PROJECTOR_JSON_OBJECT_END_REQUIRED")
    if current == ord("["):
        offset += 1
        offset = _skip_ws(data, offset)
        if offset < len(data) and data[offset] == ord("]"):
            return offset + 1
        while True:
            offset = _skip_json_value(data, offset, depth=depth + 1)
            offset = _skip_ws(data, offset)
            if offset < len(data) and data[offset] == ord(","):
                offset = _skip_ws(data, offset + 1)
                continue
            return _expect(data, offset, ord("]"), "PROJECTOR_JSON_ARRAY_END_REQUIRED")
    for literal in (b"true", b"false", b"null"):
        if data.startswith(literal, offset):
            return offset + len(literal)
    return _skip_json_number(data, offset)


def _parse_canonical_desc_id(data: bytes, offset: int) -> tuple[int, int]:
    start = offset
    while offset < len(data) and ord("0") <= data[offset] <= ord("9"):
        offset += 1
    raw = data[start:offset]
    if not raw or len(raw) > 20 or (len(raw) > 1 and raw.startswith(b"0")):
        raise IdProjectionError("PROJECTOR_DESC_ID_INVALID")
    value = int(raw.decode("ascii"))
    if value > 9_223_372_036_854_775_807:
        raise IdProjectionError("PROJECTOR_DESC_ID_OUT_OF_RANGE")
    return value, offset


def _scan_top_level(
    line: bytes,
    *,
    materialize_video_id: bool,
    audit: Optional[dict[str, int]] = None,
) -> tuple[int, Optional[str]]:
    if not isinstance(line, bytes) or not line or len(line) > MAX_LINE_BYTES:
        raise IdProjectionError("PROJECTOR_LINE_SIZE_INVALID")
    offset = _skip_ws(line, 0)
    offset = _expect(line, offset, ord("{"), "PROJECTOR_TOP_LEVEL_OBJECT_REQUIRED")
    desc_id: Optional[int] = None
    video_id: Optional[str] = None
    seen_target_fields: set[str] = set()
    offset = _skip_ws(line, offset)
    if offset < len(line) and line[offset] == ord("}"):
        raise IdProjectionError("PROJECTOR_DESC_ID_MISSING")
    while True:
        key, offset = _scan_plain_ascii_string(
            line,
            offset,
            max_bytes=64,
            code="PROJECTOR_TOP_LEVEL_KEY_INVALID",
        )
        if audit is not None:
            audit["top_level_key_semantic_decode_count"] += 1
        offset = _skip_ws(line, offset)
        offset = _expect(line, offset, ord(":"), "PROJECTOR_JSON_COLON_REQUIRED")
        offset = _skip_ws(line, offset)
        if key in {"desc_id", "vid_name"}:
            if key in seen_target_fields:
                raise IdProjectionError("PROJECTOR_TARGET_FIELD_DUPLICATE")
            seen_target_fields.add(key)
        if key == "desc_id":
            desc_id, offset = _parse_canonical_desc_id(line, offset)
            if audit is not None:
                audit["desc_id_semantic_decode_count"] += 1
        elif key == "vid_name" and materialize_video_id:
            video_id, offset = _scan_plain_ascii_string(
                line,
                offset,
                max_bytes=128,
                code="PROJECTOR_VIDEO_ID_INVALID",
            )
            if _SAFE_VIDEO_ID_RE.fullmatch(video_id) is None:
                raise IdProjectionError("PROJECTOR_VIDEO_ID_INVALID")
            if audit is not None:
                audit["video_id_semantic_decode_count"] += 1
        else:
            value_start = offset
            offset = _skip_json_value(line, offset, depth=1)
            if audit is not None:
                audit["unknown_value_syntax_bytes_scanned"] += offset - value_start
                audit["unknown_value_syntax_scan_count"] += 1
        offset = _skip_ws(line, offset)
        if offset < len(line) and line[offset] == ord(","):
            offset = _skip_ws(line, offset + 1)
            if offset < len(line) and line[offset] == ord("}"):
                raise IdProjectionError("PROJECTOR_TRAILING_COMMA_FORBIDDEN")
            continue
        offset = _expect(line, offset, ord("}"), "PROJECTOR_JSON_OBJECT_END_REQUIRED")
        break
    offset = _skip_ws(line, offset)
    if offset != len(line):
        raise IdProjectionError("PROJECTOR_TRAILING_CONTENT")
    if desc_id is None:
        raise IdProjectionError("PROJECTOR_DESC_ID_MISSING")
    if materialize_video_id and video_id is None:
        raise IdProjectionError("PROJECTOR_VIDEO_ID_MISSING")
    return desc_id, video_id


def _scan_authorized_top_level_value_slices(
    encoded_line: bytes,
    *,
    selected_fields: Sequence[str],
) -> tuple[dict[str, bytes], dict[str, int]]:
    """Syntax-check one JSONL record and retain only selected raw values."""

    if (
        type(encoded_line) is not bytes
        or not encoded_line
        or len(encoded_line) > MAX_LINE_BYTES
        or not encoded_line.endswith(b"\n")
        or b"\n" in encoded_line[:-1]
        or b"\r" in encoded_line
    ):
        raise IdProjectionError("PROJECTOR_INDEXED_LINE_BOUNDARY_INVALID")
    requested = tuple(selected_fields)
    requested_set = frozenset(requested)
    if (
        not requested
        or len(requested_set) != len(requested)
        or any(
            not isinstance(field, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", field)
            is None
            for field in requested
        )
    ):
        raise IdProjectionError("PROJECTOR_SELECTED_FIELD_SET_INVALID")

    payload = encoded_line[:-1]
    offset = _skip_ws(payload, 0)
    offset = _expect(
        payload,
        offset,
        ord("{"),
        "PROJECTOR_TOP_LEVEL_OBJECT_REQUIRED",
    )
    offset = _skip_ws(payload, offset)
    if offset < len(payload) and payload[offset] == ord("}"):
        raise IdProjectionError("PROJECTOR_SELECTED_FIELD_MISSING")

    requested_by_key_bytes = {
        field.encode("ascii", "strict"): field for field in requested
    }
    slices: dict[str, bytes] = {}
    seen_key_bytes: set[bytes] = set()
    counters = {
        "top_level_key_syntax_scan_count": 0,
        "selected_value_syntax_scan_count": 0,
        "selected_value_syntax_bytes_scanned": 0,
        "nonselected_value_syntax_scan_count": 0,
        "nonselected_value_syntax_bytes_scanned": 0,
    }
    while True:
        key_bytes, offset = _scan_plain_ascii_token_bytes(
            payload,
            offset,
            max_bytes=64,
            code="PROJECTOR_TOP_LEVEL_KEY_INVALID",
        )
        counters["top_level_key_syntax_scan_count"] += 1
        if key_bytes in seen_key_bytes:
            raise IdProjectionError("PROJECTOR_TOP_LEVEL_KEY_DUPLICATE")
        seen_key_bytes.add(key_bytes)
        offset = _skip_ws(payload, offset)
        offset = _expect(
            payload,
            offset,
            ord(":"),
            "PROJECTOR_JSON_COLON_REQUIRED",
        )
        value_start = _skip_ws(payload, offset)
        value_end = _skip_json_value(payload, value_start, depth=1)
        value_bytes = value_end - value_start
        selected_field = requested_by_key_bytes.get(key_bytes)
        if selected_field is not None:
            slices[selected_field] = payload[value_start:value_end]
            counters["selected_value_syntax_scan_count"] += 1
            counters["selected_value_syntax_bytes_scanned"] += value_bytes
        else:
            counters["nonselected_value_syntax_scan_count"] += 1
            counters["nonselected_value_syntax_bytes_scanned"] += value_bytes
        offset = _skip_ws(payload, value_end)
        if offset < len(payload) and payload[offset] == ord(","):
            offset = _skip_ws(payload, offset + 1)
            if offset < len(payload) and payload[offset] == ord("}"):
                raise IdProjectionError("PROJECTOR_TRAILING_COMMA_FORBIDDEN")
            continue
        offset = _expect(
            payload,
            offset,
            ord("}"),
            "PROJECTOR_JSON_OBJECT_END_REQUIRED",
        )
        break
    offset = _skip_ws(payload, offset)
    if offset != len(payload):
        raise IdProjectionError("PROJECTOR_TRAILING_CONTENT")
    if set(slices) != requested_set:
        raise IdProjectionError("PROJECTOR_SELECTED_FIELD_MISSING")
    return slices, counters


def _decode_selected_json_value(raw: bytes, *, code: str) -> Any:
    try:
        text = raw.decode("utf-8", "strict")
        value = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IdProjectionError(code) from error
    return value


def _decode_projection_desc_id(raw: bytes) -> int:
    value, offset = _parse_canonical_desc_id(raw, 0)
    if offset != len(raw):
        raise IdProjectionError("PROJECTOR_DESC_ID_INVALID")
    return value


def _decode_projection_query_type(raw: bytes) -> str:
    value = _decode_selected_json_value(
        raw,
        code="PROJECTOR_QUERY_TYPE_INVALID",
    )
    if not isinstance(value, str) or value not in {
        "v",
        "t",
        "vt",
        "unknown",
    }:
        raise IdProjectionError("PROJECTOR_QUERY_TYPE_INVALID")
    return value


def _decode_projection_desc(raw: bytes) -> str:
    value = _decode_selected_json_value(
        raw,
        code="PROJECTOR_QUERY_TEXT_INVALID",
    )
    if not isinstance(value, str) or "\x00" in value:
        raise IdProjectionError("PROJECTOR_QUERY_TEXT_INVALID")
    return value


def _decode_projection_timestamp(raw: bytes) -> list[float]:
    value = _decode_selected_json_value(
        raw,
        code="PROJECTOR_TIMESTAMP_INVALID",
    )
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            for item in value
        )
        or float(value[0]) < 0.0
        or float(value[1]) <= float(value[0])
    ):
        raise IdProjectionError("PROJECTOR_TIMESTAMP_INVALID")
    return [float(value[0]), float(value[1])]


def _decode_projection_video_id(raw: bytes) -> str:
    value = _decode_selected_json_value(
        raw,
        code="PROJECTOR_VIDEO_ID_INVALID",
    )
    if not isinstance(value, str) or _SAFE_VIDEO_ID_RE.fullmatch(value) is None:
        raise IdProjectionError("PROJECTOR_VIDEO_ID_INVALID")
    return value


def extract_authorized_train_projection_line(
    encoded_line: bytes,
    *,
    purpose: str,
    expected_desc_id: int,
) -> dict[str, Any]:
    """Extract one exact post-index projection without owning any source I/O.

    Non-selected values are validated only by the byte-level JSON syntax
    skipper.  ``json.loads`` is applied solely to individually selected value
    slices whose semantics are authorized by the fixed purpose.
    """

    if purpose not in AUTHORIZED_TRAIN_PROJECTION_PURPOSES:
        raise IdProjectionError("PROJECTOR_TRAIN_PURPOSE_UNAUTHORIZED")
    if (
        type(expected_desc_id) is not int
        or expected_desc_id < 0
        or expected_desc_id > 9_223_372_036_854_775_807
    ):
        raise IdProjectionError("PROJECTOR_EXPECTED_DESC_ID_INVALID")
    selected_fields = _TRAIN_PROJECTION_SOURCE_FIELDS[purpose]
    raw_values, syntax_counters = _scan_authorized_top_level_value_slices(
        encoded_line,
        selected_fields=selected_fields,
    )
    desc_id = _decode_projection_desc_id(raw_values["desc_id"])
    if desc_id != expected_desc_id:
        raise IdProjectionError("PROJECTOR_INDEX_DESC_ID_BINDING_MISMATCH")

    retained: dict[str, Any] = {"desc_id": desc_id}
    decoded_field_counts = {
        "desc_id": 1,
        "desc": 0,
        "type": 0,
        "ts": 0,
        "vid_name": 0,
    }
    raw_field_hash_counts = {"desc": 0}
    if purpose == "FORWARD_QUERY_IDENTITY":
        retained["type"] = _decode_projection_query_type(
            raw_values["type"]
        )
        retained["desc_raw_json_sha256"] = _sha256(
            raw_values["desc"]
        )
        decoded_field_counts["type"] = 1
        raw_field_hash_counts["desc"] = 1
    elif purpose == "POST_FORWARD_EVAL_STATS":
        retained["desc"] = _decode_projection_desc(raw_values["desc"])
        retained["type"] = _decode_projection_query_type(
            raw_values["type"]
        )
        retained["ts"] = _decode_projection_timestamp(raw_values["ts"])
        retained["vid_name"] = _decode_projection_video_id(
            raw_values["vid_name"]
        )
        decoded_field_counts.update(
            {"desc": 1, "type": 1, "ts": 1, "vid_name": 1}
        )
    else:
        retained["vid_name"] = _decode_projection_video_id(
            raw_values["vid_name"]
        )
        decoded_field_counts["vid_name"] = 1

    retained_fields = _TRAIN_PROJECTION_RETAINED_FIELDS[purpose]
    if set(retained) != set(retained_fields):
        raise IdProjectionError("PROJECTOR_RETAINED_FIELD_SET_DRIFT")
    counters = {
        "line_count": 1,
        "line_bytes_scanned": len(encoded_line),
        "line_boundary_verified": True,
        **syntax_counters,
        "decoded_field_counts": decoded_field_counts,
        "raw_field_hash_counts": raw_field_hash_counts,
        "full_line_json_decode_count": 0,
        "unauthorized_field_semantic_decode_count": 0,
    }
    result = {
        "schema_version": "c28f_a4_authorized_train_projection_line_v1",
        "purpose": purpose,
        "expected_desc_id": expected_desc_id,
        "retained_fields": list(retained_fields),
        "retained_record": retained,
        "counters": counters,
    }
    return result


@dataclass(frozen=True)
class ProjectionResult:
    records: Tuple[MinimalIdMapRecord, ...]
    source_line_count: int
    source_bytes_scanned: int
    expected_desc_id_count: int
    syntax_scan_pass_count: int
    top_level_key_semantic_decode_count: int
    desc_id_semantic_decode_count: int
    video_id_semantic_decode_count: int
    unknown_value_syntax_scan_count: int
    unknown_value_syntax_bytes_scanned: int
    control_capability_sha256: str
    control_binding: Mapping[str, Any]
    access_receipt: Mapping[str, Any]
    minimal_batch: Any
    control_result_seal_sha256: str
    _issuer: object
    unknown_value_semantic_decode_count: int = 0
    retained_non_id_value_count: int = 0

    def __post_init__(self) -> None:
        if self._issuer is not _PROJECTION_RESULT_ISSUER:
            raise IdProjectionError("PROJECTOR_RESULT_NOT_CONTROL_ISSUED")
        _require_sha256(self.control_capability_sha256, "control_capability_sha256")
        _require_sha256(self.control_result_seal_sha256, "control_result_seal_sha256")
        if not isinstance(self.control_binding, Mapping) or not isinstance(
            self.access_receipt, Mapping
        ):
            raise IdProjectionError("PROJECTOR_RESULT_CONTROL_BINDING_INVALID")
        try:
            from .a4_control import (
                A4G4MinimalProjectionBatch,
                validate_g4_minimal_projection_batch,
            )

            if type(self.minimal_batch) is not A4G4MinimalProjectionBatch:
                raise IdProjectionError("PROJECTOR_RESULT_MINIMAL_BATCH_INVALID")
            batch = validate_g4_minimal_projection_batch(self.minimal_batch)
        except IdProjectionError:
            raise
        except Exception as exc:
            raise IdProjectionError("PROJECTOR_RESULT_MINIMAL_BATCH_INVALID") from exc
        integer_fields = (
            "source_line_count",
            "source_bytes_scanned",
            "expected_desc_id_count",
            "syntax_scan_pass_count",
            "top_level_key_semantic_decode_count",
            "desc_id_semantic_decode_count",
            "video_id_semantic_decode_count",
            "unknown_value_syntax_scan_count",
            "unknown_value_syntax_bytes_scanned",
            "unknown_value_semantic_decode_count",
            "retained_non_id_value_count",
        )
        if any(type(getattr(self, field)) is not int or getattr(self, field) < 0 for field in integer_fields):
            raise IdProjectionError("PROJECTOR_RESULT_COUNTER_INVALID")
        if not self.records or self.expected_desc_id_count != len(self.records):
            raise IdProjectionError("PROJECTOR_RESULT_INCOMPLETE")
        if any(not isinstance(record, MinimalIdMapRecord) for record in self.records):
            raise IdProjectionError("PROJECTOR_RESULT_RECORD_TYPE_INVALID")
        desc_ids = tuple(record.desc_id for record in self.records)
        if any(
            type(desc_id) is not int
            or desc_id < 0
            or desc_id > 9_223_372_036_854_775_807
            for desc_id in desc_ids
        ):
            raise IdProjectionError("PROJECTOR_RESULT_DESC_ID_INVALID")
        if tuple(sorted(set(desc_ids))) != desc_ids:
            raise IdProjectionError("PROJECTOR_RESULT_NOT_CANONICAL")
        if any(_SAFE_VIDEO_ID_RE.fullmatch(record.video_id) is None for record in self.records):
            raise IdProjectionError("PROJECTOR_RESULT_VIDEO_ID_INVALID")
        if (
            self.source_line_count < len(self.records)
            or self.source_bytes_scanned < self.source_line_count
        ):
            raise IdProjectionError("PROJECTOR_RESULT_SOURCE_COUNT_INVALID")
        indexed_batch = (
            batch.get("schema_version")
            == "c28f_a4_g4_minimal_projection_batch_v2"
        )
        if indexed_batch:
            if (
                self.source_line_count != len(self.records)
                or self.syntax_scan_pass_count < self.source_line_count
            ):
                raise IdProjectionError(
                    "PROJECTOR_RESULT_INDEXED_SCAN_COUNT_INVALID"
                )
        elif self.syntax_scan_pass_count != self.source_line_count + len(
            self.records
        ):
            raise IdProjectionError("PROJECTOR_RESULT_SCAN_PASS_COUNT_INVALID")
        if self.video_id_semantic_decode_count != len(self.records):
            raise IdProjectionError("PROJECTOR_RESULT_VIDEO_DECODE_COUNT_INVALID")
        expected_desc_decodes = (
            len(self.records) if indexed_batch else self.syntax_scan_pass_count
        )
        if self.desc_id_semantic_decode_count != expected_desc_decodes:
            raise IdProjectionError("PROJECTOR_RESULT_DESC_DECODE_COUNT_INVALID")
        if self.unknown_value_semantic_decode_count != 0 or self.retained_non_id_value_count != 0:
            raise IdProjectionError("PROJECTOR_RESULT_NON_ID_SEMANTIC_RETENTION")
        expected_records = [
            {"desc_id": record.desc_id, "video_id": record.video_id}
            for record in self.records
        ]
        if (
            batch.get("running_capability_sha256")
            != self.control_capability_sha256
            or batch.get("records") != expected_records
            or batch.get("scanner_counters")
            != {
                "source_line_count": self.source_line_count,
                "source_bytes_scanned": self.source_bytes_scanned,
                "expected_desc_id_count": self.expected_desc_id_count,
                "syntax_scan_pass_count": self.syntax_scan_pass_count,
                "top_level_key_semantic_decode_count": self.top_level_key_semantic_decode_count,
                "desc_id_semantic_decode_count": self.desc_id_semantic_decode_count,
                "video_id_semantic_decode_count": self.video_id_semantic_decode_count,
                "unknown_value_syntax_scan_count": self.unknown_value_syntax_scan_count,
                "unknown_value_syntax_bytes_scanned": self.unknown_value_syntax_bytes_scanned,
            }
            or batch.get("access_receipt") != dict(self.access_receipt)
        ):
            raise IdProjectionError("PROJECTOR_RESULT_MINIMAL_BATCH_DRIFT")

    def audit(self) -> Mapping[str, Any]:
        return {
            "schema_version": PROJECTOR_SCHEMA_VERSION,
            "status": "CONTROL_ACCESS_CAS_COMMITTED_STAGE_CAS_PENDING",
            "source_line_count": self.source_line_count,
            "source_bytes_scanned": self.source_bytes_scanned,
            "source_content_was_read": True,
            "expected_desc_id_count": self.expected_desc_id_count,
            "projected_record_count": len(self.records),
            "syntax_scan_pass_count": self.syntax_scan_pass_count,
            "top_level_key_semantic_decode_count": self.top_level_key_semantic_decode_count,
            "desc_id_semantic_decode_count": self.desc_id_semantic_decode_count,
            "video_id_semantic_decode_count": self.video_id_semantic_decode_count,
            "unknown_value_syntax_scan_count": self.unknown_value_syntax_scan_count,
            "unknown_value_syntax_bytes_scanned": self.unknown_value_syntax_bytes_scanned,
            "unknown_value_semantic_decode_count": self.unknown_value_semantic_decode_count,
            "retained_non_id_value_count": self.retained_non_id_value_count,
            "unknown_value_presence_or_semantics_asserted": False,
            "manifest_content_open_count": 3,
            "protected_source_content_open_count": 2,
            "official_val_content_open_count": 0,
            "total_authorized_id_mapping_content_open_count": 5,
            "model_forward_count": 0,
            "evaluator_count": 0,
        }


def _scan_authorized_id_mapping_lines(
    lines: Iterable[bytes], *, expected_desc_ids: Sequence[int]
) -> Mapping[str, Any]:
    """Scan only the controller-derived protected-ID union.

    The expected IDs are supplied only by the controller after it re-reads the
    two protected Genesis manifests.  Every source row gets a syntax/desc-ID
    pass; only rows in that exact union get the second, video-ID materialising
    pass.  Non-target values never cross the scanner boundary.
    """

    expected = tuple(expected_desc_ids)
    if not expected or any(type(value) is not int or value < 0 for value in expected):
        raise IdProjectionError("PROJECTOR_EXPECTED_DESC_IDS_INVALID")
    if tuple(sorted(set(expected))) != expected:
        raise IdProjectionError("PROJECTOR_EXPECTED_DESC_IDS_NOT_CANONICAL")
    expected_set = frozenset(expected)
    projected: dict[int, MinimalIdMapRecord] = {}
    audit = {
        "top_level_key_semantic_decode_count": 0,
        "desc_id_semantic_decode_count": 0,
        "video_id_semantic_decode_count": 0,
        "unknown_value_syntax_scan_count": 0,
        "unknown_value_syntax_bytes_scanned": 0,
    }
    line_count = 0
    byte_count = 0
    for line in lines:
        line_count += 1
        if not isinstance(line, bytes):
            raise IdProjectionError("PROJECTOR_LINE_TYPE_INVALID")
        byte_count += len(line)
        desc_id, _unused = _scan_top_level(
            line,
            materialize_video_id=False,
            audit=audit,
        )
        if desc_id not in expected_set:
            continue
        if desc_id in projected:
            raise IdProjectionError("PROJECTOR_AUTHORIZED_DESC_ID_DUPLICATE")
        second_desc_id, video_id = _scan_top_level(
            line,
            materialize_video_id=True,
            audit=audit,
        )
        if second_desc_id != desc_id or video_id is None:
            raise IdProjectionError("PROJECTOR_TWO_PASS_IDENTITY_DRIFT")
        projected[desc_id] = MinimalIdMapRecord(
            desc_id=desc_id,
            video_id=video_id,
        )
    if set(projected) != expected_set:
        raise IdProjectionError("PROJECTOR_AUTHORIZED_DESC_ID_SET_INCOMPLETE")
    return {
        "records": tuple(projected[desc_id] for desc_id in expected),
        "source_line_count": line_count,
        "source_bytes_scanned": byte_count,
        "expected_desc_id_count": len(expected),
        "syntax_scan_pass_count": line_count + len(expected),
        "top_level_key_semantic_decode_count": audit[
            "top_level_key_semantic_decode_count"
        ],
        "desc_id_semantic_decode_count": audit["desc_id_semantic_decode_count"],
        "video_id_semantic_decode_count": audit["video_id_semantic_decode_count"],
        "unknown_value_syntax_scan_count": audit["unknown_value_syntax_scan_count"],
        "unknown_value_syntax_bytes_scanned": audit[
            "unknown_value_syntax_bytes_scanned"
        ],
    }


def project_authorized_id_mapping(
    control_reader: Any,
) -> ProjectionResult:
    """Consume a concrete controller-issued RUNNING descriptor exactly once."""

    try:
        from .a4_control import (
            A4G4MinimalProjectionBatch,
            A4G4RunningProjectionReader,
            validate_g4_minimal_projection_batch,
            validate_g4_running_projection_capability,
        )
    except ImportError as exc:
        raise IdProjectionError("PROJECTOR_CONCRETE_CONTROL_READER_UNAVAILABLE") from exc
    if type(control_reader) is not A4G4RunningProjectionReader:
        raise IdProjectionError("PROJECTOR_CONCRETE_CONTROL_READER_REQUIRED")
    for method_name in (
        "running_capability",
        "consume_minimal_projection_once",
        "seal_projection_result",
    ):
        if not callable(getattr(control_reader, method_name, None)):
            raise IdProjectionError("PROJECTOR_CONCRETE_CONTROL_READER_INCOMPLETE")
    running_capability = control_reader.running_capability()
    try:
        control_binding = validate_g4_running_projection_capability(
            running_capability
        )
    except Exception as exc:
        raise IdProjectionError("PROJECTOR_RUNNING_CONTROL_INVALID") from exc
    minimal_batch = control_reader.consume_minimal_projection_once()
    if type(minimal_batch) is not A4G4MinimalProjectionBatch:
        raise IdProjectionError("PROJECTOR_CONTROL_MINIMAL_BATCH_REQUIRED")
    try:
        validated_batch = validate_g4_minimal_projection_batch(minimal_batch)
        batch = minimal_batch.consume_snapshot_once()
    except Exception as exc:
        raise IdProjectionError("PROJECTOR_CONTROL_MINIMAL_BATCH_INVALID") from exc
    if batch != validated_batch:
        raise IdProjectionError("PROJECTOR_CONTROL_MINIMAL_BATCH_DRIFT")
    records = tuple(
        MinimalIdMapRecord(
            desc_id=record["desc_id"],
            video_id=record["video_id"],
        )
        for record in batch["records"]
    )
    scanned = {
        "records": records,
        **dict(batch["scanner_counters"]),
    }
    access_receipt = dict(batch["access_receipt"])
    access_sha = _require_sha256(
        access_receipt.get("receipt_sha256"), "access receipt_sha256"
    )
    if access_sha != _semantic_sha256(access_receipt, ("receipt_sha256",)):
        raise IdProjectionError("PROJECTOR_CONTROL_ACCESS_RECEIPT_INVALID")
    result_payload_sha256 = _sha256(
        _canonical_json_bytes(
            {
                "records": [record.as_dict() for record in scanned["records"]],
                "counters": {
                    key: value for key, value in scanned.items() if key != "records"
                },
                "control_capability_sha256": running_capability.capability_sha256,
                "minimal_batch_capability_sha256": minimal_batch.capability_sha256,
                "minimal_batch_binding_sha256": batch["batch_binding_sha256"],
                "access_receipt_sha256": access_sha,
            }
        )
    )
    control_result_seal_sha256 = _require_sha256(
        control_reader.seal_projection_result(
            batch_capability_sha256=minimal_batch.capability_sha256,
            result_payload_sha256=result_payload_sha256
        ),
        "control_result_seal_sha256",
    )
    return ProjectionResult(
        **scanned,
        control_capability_sha256=running_capability.capability_sha256,
        control_binding=dict(control_binding),
        access_receipt=access_receipt,
        minimal_batch=minimal_batch,
        control_result_seal_sha256=control_result_seal_sha256,
        _issuer=_PROJECTION_RESULT_ISSUER,
    )


def build_projection_manifest(result: ProjectionResult) -> dict[str, Any]:
    """Build only from a scanner result sealed by the concrete control reader."""

    if (
        type(result) is not ProjectionResult
        or result._issuer is not _PROJECTION_RESULT_ISSUER
    ):
        raise IdProjectionError("PROJECTOR_CONTROL_SEALED_RESULT_REQUIRED")
    try:
        from .a4_control import validate_g4_minimal_projection_batch

        minimal_batch = validate_g4_minimal_projection_batch(
            result.minimal_batch
        )
    except Exception as exc:
        raise IdProjectionError("PROJECTION_MINIMAL_BATCH_INVALID") from exc
    binding = dict(
        _require_exact_mapping(
            result.control_binding,
            (
                "schema_version",
                "running_anchor",
                "protected_manifest_contract",
                "single_use",
                "capability_binding_sha256",
            ),
            "PROJECTION_CONTROL_BINDING_FIELDS_MISMATCH",
        )
    )
    if (
        binding.get("schema_version")
        != "c28f_a4_g4_running_projection_capability_v1"
        or binding.get("single_use") is not True
    ):
        raise IdProjectionError("PROJECTION_RUNNING_CONTROL_BINDING_INVALID")
    capability_binding_sha = _require_sha256(
        binding.get("capability_binding_sha256"),
        "capability_binding_sha256",
    )
    if capability_binding_sha != _semantic_sha256(
        binding,
        ("capability_binding_sha256",),
    ):
        raise IdProjectionError("PROJECTION_CONTROL_CAPABILITY_BINDING_HASH_MISMATCH")
    protected_manifest_contract = dict(
        _require_exact_mapping(
            binding.get("protected_manifest_contract"),
            (
                "schema_version",
                "manifests",
                "union_count",
                "union_sha256",
                "selection_rule",
                "contract_sha256",
            ),
            "PROJECTION_PROTECTED_MANIFEST_CONTRACT_FIELDS_MISMATCH",
        )
    )
    protected_manifest_contract_sha = _require_sha256(
        protected_manifest_contract.get("contract_sha256"),
        "protected manifest contract SHA",
    )
    if (
        protected_manifest_contract_sha
        != _semantic_sha256(
            protected_manifest_contract,
            ("contract_sha256",),
        )
        or protected_manifest_contract_sha
        != minimal_batch.get("protected_manifest_contract_sha256")
        or protected_manifest_contract.get("union_sha256")
        != minimal_batch.get("expected_desc_ids_sha256")
        or protected_manifest_contract.get("union_count")
        != len(minimal_batch.get("expected_desc_ids", []))
    ):
        raise IdProjectionError("PROJECTION_PROTECTED_MANIFEST_CONTRACT_DRIFT")
    try:
        running = dict(
            validate_protected_id_mapping_running_anchor(
                binding.get("running_anchor")
            )
        )
    except Exception as exc:
        raise IdProjectionError("PROJECTION_RUNNING_ANCHOR_INVALID") from exc
    if (
        running.get("schema_version")
        != PROTECTED_ID_MAPPING_RUNNING_ANCHOR_SCHEMA_VERSION
        or any(
            forbidden in running
            for forbidden in ("run_id", "token_id", "eval_id")
        )
    ):
        raise IdProjectionError("PROJECTION_RUNNING_ANCHOR_SCHEMA_MISMATCH")

    access = dict(
        _require_exact_mapping(
            result.access_receipt,
            (
                "schema_version",
                "status",
                "goal_id",
                "attempt_id",
                "authority_permission_id",
                "operation_id",
                "access_transaction_id",
                "running_state_sha256",
                "running_event_sha256",
                "control_capability_sha256",
                "descriptor_identity_sha256",
                "access_intent_sha256",
                "source_line_count",
                "source_bytes_read",
                "syntax_scan_pass_count",
                "unknown_value_syntax_bytes_scanned",
                "unknown_value_semantic_decode_count",
                "retained_non_id_value_count",
                "train_fit_source_line_count",
                "train_fit_source_bytes_read",
                "train_fit_syntax_scan_pass_count",
                "train_fit_unknown_value_syntax_bytes_scanned",
                "manifest_content_open_count",
                "protected_source_content_open_count",
                "official_val_content_open_count",
                "source_observation_receipt_sha256",
                "content_open_count",
                "protected_id_mapping_operation_count",
                "model_forward_count",
                "evaluator_count",
                "unmediated_content_open_count",
                "instrumentation_complete",
                "eval_token_consumed",
                "operation_state_before",
                "operation_state_after",
                "access_state_sha256",
                "access_event_sha256",
                "receipt_sha256",
            ),
            "PROJECTION_CONTROL_ACCESS_RECEIPT_FIELDS_MISMATCH",
        )
    )
    access_sha = _require_sha256(access["receipt_sha256"], "receipt_sha256")
    if access_sha != _semantic_sha256(access, ("receipt_sha256",)):
        raise IdProjectionError("PROJECTION_CONTROL_ACCESS_RECEIPT_SELF_HASH_MISMATCH")
    expected_access = {
        "schema_version": MAPPING_ACCESS_RECEIPT_SCHEMA_VERSION,
        "status": "FSYNCED_ID_ONLY_ACCESS_COMMITTED",
        "goal_id": running["goal_id"],
        "attempt_id": running["attempt_id"],
        "authority_permission_id": running["authority_permission_id"],
        "operation_id": running["operation_id"],
        "control_capability_sha256": result.control_capability_sha256,
        "descriptor_identity_sha256": running["descriptor_identity_sha256"],
        "access_intent_sha256": running["access_intent_sha256"],
        "source_line_count": result.source_line_count,
        "source_bytes_read": result.source_bytes_scanned,
        "syntax_scan_pass_count": result.syntax_scan_pass_count,
        "unknown_value_syntax_bytes_scanned": result.unknown_value_syntax_bytes_scanned,
        "unknown_value_semantic_decode_count": 0,
        "retained_non_id_value_count": 0,
        "train_fit_source_line_count": minimal_batch[
            "train_fit_source_evidence"
        ]["record_count"],
        "train_fit_source_bytes_read": minimal_batch[
            "train_fit_source_evidence"
        ]["train_projection_access_receipt"]["pread_byte_count"],
        "train_fit_syntax_scan_pass_count": minimal_batch[
            "train_fit_source_evidence"
        ]["train_projection_access_receipt"]
        ["top_level_key_syntax_scan_count"],
        "train_fit_unknown_value_syntax_bytes_scanned": minimal_batch[
            "train_fit_source_evidence"
        ]["train_projection_access_receipt"]
        ["nonselected_value_syntax_bytes_scanned"],
        "manifest_content_open_count": 3,
        "protected_source_content_open_count": 2,
        "official_val_content_open_count": 0,
        "source_observation_receipt_sha256": minimal_batch[
            "source_observation_receipt"
        ]["receipt_sha256"],
        "content_open_count": 5,
        "protected_id_mapping_operation_count": 1,
        "model_forward_count": 0,
        "evaluator_count": 0,
        "unmediated_content_open_count": 0,
        "instrumentation_complete": True,
        "eval_token_consumed": False,
        "operation_state_before": "RUNNING",
        "operation_state_after": "ACCESSED_AWAITING_PROJECTION_CAS",
        "running_state_sha256": running["running_state_sha256"],
        "running_event_sha256": running["running_event_sha256"],
    }
    if any(access.get(field) != value for field, value in expected_access.items()):
        raise IdProjectionError("PROJECTION_CONTROL_ACCESS_RECEIPT_BINDING_MISMATCH")
    if not isinstance(access.get("access_transaction_id"), str) or not access[
        "access_transaction_id"
    ]:
        raise IdProjectionError("PROJECTION_ACCESS_TRANSACTION_MISSING")
    for field in ("access_state_sha256", "access_event_sha256"):
        _require_sha256(access.get(field), field)

    mapping_rows = [
        {"desc_id": record.desc_id, "video_id": record.video_id}
        for record in result.records
    ]
    video_ids = sorted({record.video_id for record in result.records})
    lineage = {
        "goal_id": running["goal_id"],
        "attempt_id": running["attempt_id"],
        "authority_permission_id": running["authority_permission_id"],
        "operation_id": running["operation_id"],
        "operation_transaction_id": running["transaction_id"],
        "parent_state_sha256": running["parent_state_sha256"],
        "parent_event_sha256": running["parent_event_sha256"],
        "running_state_sha256": running["running_state_sha256"],
        "running_event_sha256": running["running_event_sha256"],
        "access_transaction_id": access["access_transaction_id"],
        "access_state_sha256": access["access_state_sha256"],
        "access_event_sha256": access["access_event_sha256"],
        "descriptor_identity_sha256": running["descriptor_identity_sha256"],
        "access_intent_sha256": running["access_intent_sha256"],
        "authority_source_file_sha256": running[
            "authority_source_file_sha256"
        ],
        "protected_manifest_contract_sha256": protected_manifest_contract_sha,
        "expected_desc_ids_sha256": minimal_batch[
            "expected_desc_ids_sha256"
        ],
        "minimal_projection_batch_capability_sha256": (
            result.minimal_batch.capability_sha256
        ),
        "minimal_projection_batch_binding_sha256": minimal_batch[
            "batch_binding_sha256"
        ],
        "minimal_projection_records_sha256": minimal_batch[
            "records_sha256"
        ],
        "scanner_contract_sha256": minimal_batch["scanner_contract_sha256"],
        "source_observation_receipt_sha256": minimal_batch[
            "source_observation_receipt"
        ]["receipt_sha256"],
        "running_anchor_binding_sha256": running["anchor_binding_sha256"],
        "control_capability_sha256": result.control_capability_sha256,
        "control_capability_binding_sha256": capability_binding_sha,
        "access_ledger_receipt_sha256": access_sha,
        "control_result_seal_sha256": result.control_result_seal_sha256,
        "eval_token_consumed": False,
    }
    base = {
        "schema_version": PROJECTION_MANIFEST_SCHEMA_VERSION,
        "status": "PROJECTED_CANDIDATE_ACCESS_CAS_COMMITTED_STAGE_CAS_PENDING",
        "authority_scope": "DESC_ID_TO_VIDEO_ID_ONLY",
        "lineage": lineage,
        "mapping_rows": mapping_rows,
        "mapping_rows_sha256": _sha256(_canonical_json_bytes(mapping_rows)),
        "protected_video_ids": video_ids,
        "protected_video_id_count": len(video_ids),
        "protected_video_id_lines_sha256": _sha256(
            "".join(f"{value}\n" for value in video_ids).encode("utf-8")
        ),
        "projection_audit": dict(result.audit()),
        "unknown_value_semantics": {
            "presence_asserted": False,
            "semantic_decode_count": result.unknown_value_semantic_decode_count,
            "retained_or_output_count": result.retained_non_id_value_count,
            "syntax_bytes_scanned": result.unknown_value_syntax_bytes_scanned,
        },
        "content_open_classification": {
            "categories_mutually_exclusive": True,
            "unauthorized_or_non_id_only_protected_content_open_count": 0,
            "authorized_protected_id_mapping_content_open_count": 5,
            "total_protected_content_open_count": 5,
            "authorized_id_mapping_breakdown": {
                "protected_desc_id_manifest_content_open_count": 2,
                "train_fit_desc_id_manifest_content_open_count": 1,
                "protected_desc_train_jsonl_indexed_pread_content_open_count": 1,
                "train_fit_train_jsonl_indexed_pread_content_open_count": 1,
                "official_val_content_open_count": 0,
            },
        },
        "completion_status": (
            "DEFERRED_REQUIRES_A4_STAGE_CAS_AND_CONTROL_COMPLETION_SIDECAR"
        ),
    }
    return {**base, "manifest_sha256": _semantic_sha256(base)}


def build_projection_commit_request(
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Prepare a non-authoritative request for the reviewed G4 stage builder.

    This pure request cannot complete, consume, or grant a token. Completion
    remains unavailable until the exact five-file StageBundle is committed
    and the controller writes and re-reads its completion sidecar.
    """

    current = dict(manifest)
    manifest_sha = _require_sha256(current.get("manifest_sha256"), "manifest_sha256")
    if manifest_sha != _semantic_sha256(current, ("manifest_sha256",)):
        raise IdProjectionError("PROJECTION_MANIFEST_SELF_HASH_MISMATCH")
    if (
        current.get("schema_version") != PROJECTION_MANIFEST_SCHEMA_VERSION
        or current.get("status")
        != "PROJECTED_CANDIDATE_ACCESS_CAS_COMMITTED_STAGE_CAS_PENDING"
    ):
        raise IdProjectionError("PROJECTION_MANIFEST_NOT_COMMITTABLE")
    lineage = dict(
        _require_exact_mapping(
            current.get("lineage"),
            (
                "goal_id",
                "attempt_id",
                "authority_permission_id",
                "operation_id",
                "operation_transaction_id",
                "parent_state_sha256",
                "parent_event_sha256",
                "running_state_sha256",
                "running_event_sha256",
                "access_transaction_id",
                "access_state_sha256",
                "access_event_sha256",
                "descriptor_identity_sha256",
                "access_intent_sha256",
                "authority_source_file_sha256",
                "protected_manifest_contract_sha256",
                "expected_desc_ids_sha256",
                "minimal_projection_batch_capability_sha256",
                "minimal_projection_batch_binding_sha256",
                "minimal_projection_records_sha256",
                "scanner_contract_sha256",
                "source_observation_receipt_sha256",
                "running_anchor_binding_sha256",
                "control_capability_sha256",
                "control_capability_binding_sha256",
                "access_ledger_receipt_sha256",
                "control_result_seal_sha256",
                "eval_token_consumed",
            ),
            "PROJECTION_MANIFEST_LINEAGE_FIELDS_MISMATCH",
        )
    )
    for field in (
        "parent_state_sha256",
        "parent_event_sha256",
        "running_state_sha256",
        "running_event_sha256",
        "access_state_sha256",
        "access_event_sha256",
        "descriptor_identity_sha256",
        "access_intent_sha256",
        "authority_source_file_sha256",
        "protected_manifest_contract_sha256",
        "expected_desc_ids_sha256",
        "minimal_projection_batch_capability_sha256",
        "minimal_projection_batch_binding_sha256",
        "minimal_projection_records_sha256",
        "scanner_contract_sha256",
        "source_observation_receipt_sha256",
        "running_anchor_binding_sha256",
        "control_capability_sha256",
        "control_capability_binding_sha256",
        "access_ledger_receipt_sha256",
        "control_result_seal_sha256",
    ):
        _require_sha256(lineage.get(field), field)
    if (
        lineage.get("authority_permission_id")
        != "AUTH_PROTECTED_ID_MAPPING_ONLY"
        or lineage.get("eval_token_consumed") is not False
        or not isinstance(lineage.get("access_transaction_id"), str)
        or not lineage["access_transaction_id"]
    ):
        raise IdProjectionError("PROJECTION_MANIFEST_AUTHORITY_SCOPE_MISMATCH")
    base = {
        "schema_version": PROJECTION_COMMIT_REQUEST_SCHEMA_VERSION,
        "status": "DEFERRED_REQUIRES_REVIEWED_A4_STAGE_BUNDLE_CAS",
        "goal_id": lineage["goal_id"],
        "attempt_id": lineage["attempt_id"],
        "authority_permission_id": lineage["authority_permission_id"],
        "operation_id": lineage["operation_id"],
        "operation_transaction_id": lineage["operation_transaction_id"],
        "parent_state_sha256": lineage["parent_state_sha256"],
        "parent_event_sha256": lineage["parent_event_sha256"],
        "running_state_sha256": lineage["running_state_sha256"],
        "running_event_sha256": lineage["running_event_sha256"],
        "access_transaction_id": lineage["access_transaction_id"],
        "access_state_sha256": lineage["access_state_sha256"],
        "access_event_sha256": lineage["access_event_sha256"],
        "descriptor_identity_sha256": lineage["descriptor_identity_sha256"],
        "access_intent_sha256": lineage["access_intent_sha256"],
        "authority_source_file_sha256": lineage[
            "authority_source_file_sha256"
        ],
        "protected_manifest_contract_sha256": lineage[
            "protected_manifest_contract_sha256"
        ],
        "expected_desc_ids_sha256": lineage["expected_desc_ids_sha256"],
        "minimal_projection_batch_capability_sha256": lineage[
            "minimal_projection_batch_capability_sha256"
        ],
        "minimal_projection_batch_binding_sha256": lineage[
            "minimal_projection_batch_binding_sha256"
        ],
        "minimal_projection_records_sha256": lineage[
            "minimal_projection_records_sha256"
        ],
        "scanner_contract_sha256": lineage["scanner_contract_sha256"],
        "source_observation_receipt_sha256": lineage[
            "source_observation_receipt_sha256"
        ],
        "running_anchor_binding_sha256": lineage[
            "running_anchor_binding_sha256"
        ],
        "control_capability_sha256": lineage["control_capability_sha256"],
        "control_capability_binding_sha256": lineage[
            "control_capability_binding_sha256"
        ],
        "access_ledger_receipt_sha256": lineage[
            "access_ledger_receipt_sha256"
        ],
        "control_result_seal_sha256": lineage["control_result_seal_sha256"],
        "projection_manifest_sha256": manifest_sha,
        "pure_function_claims_persistence": False,
        "model_forward_allowed": False,
        "eval_token_consumed": False,
        "completion_available": False,
    }
    return {**base, "request_sha256": _semantic_sha256(base)}


__all__ = [
    "AUTHORIZED_TRAIN_PROJECTION_PURPOSES",
    "IdProjectionError",
    "ProjectionResult",
    "build_projection_commit_request",
    "build_projection_manifest",
    "extract_authorized_train_projection_line",
    "project_authorized_id_mapping",
]
