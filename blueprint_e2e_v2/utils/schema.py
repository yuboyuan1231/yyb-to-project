from __future__ import annotations

INFERENCE_FORBIDDEN_COLUMNS = {
    "gt_video_id",
    "gt_start",
    "gt_end",
    "candidate_iou",
    "iou",
    "iou_ge_05",
    "iou_ge_07",
    "correct_video",
    "oracle_span_score",
}


def assert_no_inference_forbidden(columns: list[str] | tuple[str, ...]) -> None:
    bad = sorted(set(columns) & INFERENCE_FORBIDDEN_COLUMNS)
    if bad:
        raise ValueError(f"forbidden inference columns present: {bad}")


def dataset_schema() -> list[str]:
    return [
        "query_id",
        "video_id",
        "split",
        "candidate_rank",
        "span_start",
        "span_end",
        "span_duration",
        "proposal_source",
        "proposal_rank",
        "dynamic_retriever_score",
        "schema_hash",
        "config_hash",
    ]

