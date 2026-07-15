"""Pure C28F metric contract and non-opening C7 replay adapter.

This module deliberately owns no filesystem paths, authorities, factories, or file
opens.  Its C7 helper only aligns iterables already opened by the concrete G2
control handle after a persisted one-shot access receipt.  The G1 firewall and
stage controller remain the sole access authority.

All public rates use ratio units in ``[0, 1]``.  Video ranks are one-based.  A
missing ground-truth video is an explicit recall miss, never a skipped query.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


METRIC_SCHEMA_VERSION = "c28f_metrics_v1"
GATE_SCHEMA_VERSION = "c28f_gate_schema_v1"
FIXTURE_EXPECTED_SCHEMA_VERSION = "c28f_metric_fixture_expected_v1"
UNIT = "ratio"

POOLED_TRUE_VR_KS = (1, 5, 10, 100, 200, 500, 1000)
LATE_TRUE_VR_KS = (1, 5, 10, 50, 100, 200)
FINAL_VIDEO_TRUE_VR_KS = (1, 5, 10, 50, 100, 200)
JOINT_MARGINAL_VR_KS = (1, 5, 10, 50, 100, 200)
LEGACY_UNIQUE_VIDEO_VR_KS = (1, 5, 10, 100)
VCMR_KS = (1, 5, 10, 100)
VCMR_IOUS = (0.5, 0.7)
C7_LEGACY_VCMR_PARITY_FIELD_MAP = {
    f"{threshold:.1f}-r{k}": f"VCMR_R@{k}_IoU@{threshold:.1f}"
    for threshold in VCMR_IOUS
    for k in VCMR_KS
}

G_BROAD_THRESHOLD = 0.90
G_KEEP_THRESHOLD = 0.90
G_FRONT_THRESHOLD = 0.80
G_LATE_PRODUCT_TOLERANCE = 1e-8
QUERY_COMPLETENESS_THRESHOLD = 1.0
HISTORICAL_PARITY_TOLERANCE = 1e-6
HIGH_CONFIDENCE_QUANTILE = 0.90
JOINT_MASS_TOLERANCE = 1e-8
FROZEN_NMS_IOU_THRESHOLD = 0.7
FROZEN_NMS_MAX_KEEP = 100
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")

REQUIRED_QUERY_FIELDS = (
    "query_id",
    "gt_video_id",
    "gt_span_sec",
    "pooled_video_order",
    "late_video_order",
    "final_video_order",
    "joint_proposals",
)


class MetricContractError(ValueError):
    """Raised when an input cannot be interpreted without changing the contract."""


def _is_real_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MetricContractError("%s must be a mapping" % field)
    return value


def _require_video_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise MetricContractError("%s must be a canonical safe identifier" % field)
    return value


def _require_query_id(value: Any, field: str = "query_id") -> Any:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, str)):
        raise MetricContractError("%s must be a non-null integer or string" % field)
    if isinstance(value, str) and _SAFE_ID_RE.fullmatch(value) is None:
        raise MetricContractError("%s must be a canonical safe identifier" % field)
    return value


def _query_identity(query_id: Any) -> Tuple[str, Any]:
    return (type(query_id).__name__, query_id)


def _assert_expected_query_universe(
    observed_query_ids: Sequence[Any], expected_query_ids: Optional[Sequence[Any]]
) -> None:
    if expected_query_ids is None:
        return
    if not isinstance(expected_query_ids, (list, tuple)) or not expected_query_ids:
        raise MetricContractError("expected_query_ids must be a non-empty explicit sequence")
    expected = [_require_query_id(value, "expected_query_ids") for value in expected_query_ids]
    expected_identities = [_query_identity(value) for value in expected]
    observed_identities = [_query_identity(value) for value in observed_query_ids]
    if len(expected_identities) != len(set(expected_identities)):
        raise MetricContractError("expected query universe contains duplicate IDs")
    if len(observed_identities) != len(set(observed_identities)):
        raise MetricContractError("observed query universe contains duplicate IDs")
    if set(observed_identities) != set(expected_identities):
        raise MetricContractError("observed query universe differs from the authorized manifest")


def _require_span(value: Any, field: str) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise MetricContractError("%s must be [start_sec, end_sec]" % field)
    start, end = value
    if not _is_real_number(start) or not _is_real_number(end):
        raise MetricContractError("%s endpoints must be finite numbers" % field)
    start_f, end_f = float(start), float(end)
    if start_f < 0.0 or end_f <= start_f:
        raise MetricContractError("%s must satisfy 0 <= start < end" % field)
    return (start_f, end_f)


def temporal_iou(left: Sequence[float], right: Sequence[float]) -> float:
    """Return continuous-time temporal IoU for two validated or raw spans."""

    left_start, left_end = _require_span(left, "left_span")
    right_start, right_end = _require_span(right, "right_span")
    intersection = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    union = (left_end - left_start) + (right_end - right_start) - intersection
    if union <= 0.0:
        raise MetricContractError("temporal IoU union must be positive")
    return intersection / union


@dataclass(frozen=True)
class _RankedVideo:
    video_id: str
    score: Optional[float]


@dataclass(frozen=True)
class _JointProposal:
    video_id: str
    start_sec: float
    end_sec: float
    score: float
    probability: Optional[float]
    ordinal: int


def _normalise_video_ranking(value: Any, field: str) -> List[_RankedVideo]:
    if not isinstance(value, (list, tuple)):
        raise MetricContractError("%s must be an explicit ordered sequence" % field)
    ranked: List[_RankedVideo] = []
    scored_mode: Optional[bool] = None
    for index, item in enumerate(value):
        if isinstance(item, str):
            if scored_mode is True:
                raise MetricContractError("%s cannot mix scored rows and video IDs" % field)
            scored_mode = False
            ranked.append(_RankedVideo(_require_video_id(item, "%s[%d]" % (field, index)), None))
            continue
        if isinstance(item, Mapping):
            if scored_mode is False:
                raise MetricContractError("%s cannot mix scored rows and video IDs" % field)
            scored_mode = True
            video_id = _require_video_id(item.get("video_id"), "%s[%d].video_id" % (field, index))
            score = item.get("score")
            if not _is_real_number(score):
                raise MetricContractError("%s[%d].score must be finite" % (field, index))
            ranked.append(_RankedVideo(video_id, float(score)))
            continue
        raise MetricContractError("%s[%d] must be a video ID or scored row" % (field, index))

    if scored_mode:
        ranked.sort(key=lambda row: (-float(row.score), row.video_id))
    video_ids = [row.video_id for row in ranked]
    if len(video_ids) != len(set(video_ids)):
        raise MetricContractError("%s contains duplicate video IDs" % field)
    return ranked


def _normalise_joint_proposals(
    value: Any,
    field: str,
    require_probabilities: bool,
    *,
    preserve_input_order: bool = False,
) -> List[_JointProposal]:
    if not isinstance(value, (list, tuple)):
        raise MetricContractError("%s must be an explicit sequence" % field)
    proposals: List[_JointProposal] = []
    for index, raw in enumerate(value):
        if isinstance(raw, Mapping):
            video_id = raw.get("video_id")
            start_sec = raw.get("start_sec")
            end_sec = raw.get("end_sec")
            score = raw.get("score")
            probability = raw.get("probability")
        elif isinstance(raw, (list, tuple)) and len(raw) in (4, 5):
            video_id, start_sec, end_sec, score = raw[:4]
            probability = raw[4] if len(raw) == 5 else None
        else:
            raise MetricContractError(
                "%s[%d] must be [video_id,start,end,score[,probability]]" % (field, index)
            )
        video = _require_video_id(video_id, "%s[%d].video_id" % (field, index))
        span = _require_span((start_sec, end_sec), "%s[%d].span" % (field, index))
        if not _is_real_number(score):
            raise MetricContractError("%s[%d].score must be finite" % (field, index))
        probability_f: Optional[float]
        if probability is None:
            probability_f = None
        elif not _is_real_number(probability) or float(probability) < 0.0:
            raise MetricContractError("%s[%d].probability must be finite and non-negative" % (field, index))
        else:
            probability_f = float(probability)
        if require_probabilities and probability_f is None:
            raise MetricContractError("%s[%d].probability is required" % (field, index))
        proposals.append(
            _JointProposal(video, span[0], span[1], float(score), probability_f, index)
        )

    if not preserve_input_order:
        proposals.sort(
            key=lambda row: (-row.score, row.video_id, row.start_sec, row.end_sec, row.ordinal)
        )
    if require_probabilities and proposals:
        total_mass = math.fsum(float(row.probability) for row in proposals)
        if not math.isclose(total_mass, 1.0, rel_tol=0.0, abs_tol=JOINT_MASS_TOLERANCE):
            raise MetricContractError(
                "%s probability mass must sum to 1 within %.1e (got %.17g)"
                % (field, JOINT_MASS_TOLERANCE, total_mass)
            )
    return proposals


def _stable_frozen_nms(proposals: Sequence[_JointProposal]) -> List[_JointProposal]:
    """Apply the frozen per-video NMS to an already stable raw score order.

    Raw ``P_full`` rows remain untouched for marginal/error-mass metrics.  Only
    this derived list is consumed by VCMR, legacy unique-video, and top1 errors.
    """

    kept: List[_JointProposal] = []
    by_video: Dict[str, List[_JointProposal]] = {}
    for proposal in proposals:
        suppress = any(
            temporal_iou(
                (proposal.start_sec, proposal.end_sec),
                (prior.start_sec, prior.end_sec),
            )
            > FROZEN_NMS_IOU_THRESHOLD
            for prior in by_video.get(proposal.video_id, [])
        )
        if suppress:
            continue
        kept.append(proposal)
        by_video.setdefault(proposal.video_id, []).append(proposal)
        if len(kept) >= FROZEN_NMS_MAX_KEEP:
            break
    return kept


def _rank_of(ranking: Sequence[_RankedVideo], gt_video_id: str) -> Optional[int]:
    for index, row in enumerate(ranking):
        if row.video_id == gt_video_id:
            return index + 1
    return None


def _rank_in_ids(video_ids: Sequence[str], gt_video_id: str) -> Optional[int]:
    for index, video_id in enumerate(video_ids):
        if video_id == gt_video_id:
            return index + 1
    return None


def _recall(ranks: Sequence[Optional[int]], k: int) -> float:
    if not ranks:
        raise MetricContractError("recall denominator must not be zero")
    return sum(rank is not None and rank <= k for rank in ranks) / float(len(ranks))


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return math.fsum(values) / float(len(values))


def _median_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def _ranking_margins(
    ranking: Sequence[_RankedVideo], gt_video_id: str
) -> Tuple[Optional[float], Optional[float]]:
    if not ranking or any(row.score is None for row in ranking):
        return (None, None)
    top_margin = None
    if len(ranking) >= 2:
        top_margin = float(ranking[0].score) - float(ranking[1].score)
    gt_score = None
    best_wrong_score = None
    for row in ranking:
        if row.video_id == gt_video_id:
            gt_score = float(row.score)
        elif best_wrong_score is None:
            best_wrong_score = float(row.score)
    gt_wrong_margin = None
    if gt_score is not None and best_wrong_score is not None:
        gt_wrong_margin = gt_score - best_wrong_score
    return (top_margin, gt_wrong_margin)


def _validate_retrieval_query(query: Mapping[str, Any], index: int) -> Dict[str, Any]:
    # Access fields explicitly.  Do not iterate over the mapping: this function is
    # intentionally incapable of consulting joint_proposals or NMS results.
    query_id = _require_query_id(query["query_id"], "queries[%d].query_id" % index)
    gt_video_id = _require_video_id(query["gt_video_id"], "queries[%d].gt_video_id" % index)
    pooled = _normalise_video_ranking(
        query["pooled_video_order"], "queries[%d].pooled_video_order" % index
    )
    late = _normalise_video_ranking(
        query["late_video_order"], "queries[%d].late_video_order" % index
    )
    final = _normalise_video_ranking(
        query["final_video_order"], "queries[%d].final_video_order" % index
    )
    pooled_ids = {row.video_id for row in pooled}
    late_ids = {row.video_id for row in late}
    final_ids = {row.video_id for row in final}
    if not late_ids.issubset(pooled_ids):
        raise MetricContractError(
            "queries[%d].late_video_order must be a subset of pooled_video_order" % index
        )
    if not final_ids.issubset(late_ids):
        raise MetricContractError(
            "queries[%d].final_video_order must be a subset of late_video_order" % index
        )
    pooled_rank = _rank_of(pooled, gt_video_id)
    late_rank = (
        _rank_of(late, gt_video_id)
        if pooled_rank is not None and pooled_rank <= max(POOLED_TRUE_VR_KS)
        else None
    )
    final_rank = _rank_of(final, gt_video_id)
    pooled_margins = _ranking_margins(pooled, gt_video_id)
    late_margins = _ranking_margins(late, gt_video_id)
    final_margins = _ranking_margins(final, gt_video_id)
    return {
        "query_id": query_id,
        "gt_video_id": gt_video_id,
        "pooled_gt_rank": pooled_rank,
        "late_gt_rank": late_rank,
        "final_video_gt_rank": final_rank,
        "pooled_to_late_rank_delta": (
            pooled_rank - late_rank
            if pooled_rank is not None and late_rank is not None
            else None
        ),
        "broad_hit_at_1000": pooled_rank is not None and pooled_rank <= 1000,
        "late_hit_at_200": late_rank is not None and late_rank <= 200,
        "pooled_top1_top2_score_margin": pooled_margins[0],
        "pooled_gt_vs_best_wrong_score_margin": pooled_margins[1],
        "late_top1_top2_score_margin": late_margins[0],
        "late_gt_vs_best_wrong_score_margin": late_margins[1],
        "final_top1_top2_score_margin": final_margins[0],
        "final_gt_vs_best_wrong_score_margin": final_margins[1],
    }


def _margin_summary(rows: Sequence[Mapping[str, Any]], prefix: str) -> Dict[str, Any]:
    top_key = "%s_top1_top2_score_margin" % prefix
    gt_key = "%s_gt_vs_best_wrong_score_margin" % prefix
    top_values = [float(row[top_key]) for row in rows if row[top_key] is not None]
    gt_values = [float(row[gt_key]) for row in rows if row[gt_key] is not None]
    if not top_values and not gt_values:
        return {
            "status": "unavailable",
            "reason": "RANKING_SCORES_NOT_PROVIDED",
            "top1_top2_available_query_count": 0,
            "gt_vs_best_wrong_available_query_count": 0,
            "mean_top1_top2_score_margin": None,
            "mean_gt_vs_best_wrong_score_margin": None,
        }
    return {
        "status": "available",
        "reason": None,
        "top1_top2_available_query_count": len(top_values),
        "gt_vs_best_wrong_available_query_count": len(gt_values),
        "mean_top1_top2_score_margin": _mean_or_none(top_values),
        "mean_gt_vs_best_wrong_score_margin": _mean_or_none(gt_values),
    }


def evaluate_true_video_retrieval(
    queries: Sequence[Mapping[str, Any]],
    *,
    expected_query_ids: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Evaluate proposal-independent video retrieval.

    The function reads only query identity, GT video, and the three video-ranking
    fields.  It has no proposal/NMS parameter and never reads ``joint_proposals``.
    """

    if not isinstance(queries, (list, tuple)) or not queries:
        raise MetricContractError("queries must be a non-empty explicit sequence")
    rows: List[Dict[str, Any]] = []
    seen_query_ids = set()
    for index, query in enumerate(queries):
        if not isinstance(query, Mapping):
            raise MetricContractError("queries[%d] must be a mapping" % index)
        try:
            row = _validate_retrieval_query(query, index)
        except KeyError as error:
            raise MetricContractError(
                "queries[%d] is missing required retrieval field %s" % (index, error.args[0])
            ) from None
        query_identity = (type(row["query_id"]).__name__, row["query_id"])
        if query_identity in seen_query_ids:
            raise MetricContractError("query_id values must be unique with type preserved")
        seen_query_ids.add(query_identity)
        rows.append(row)
    _assert_expected_query_universe(
        [row["query_id"] for row in rows], expected_query_ids
    )

    pooled_ranks = [row["pooled_gt_rank"] for row in rows]
    late_ranks = [row["late_gt_rank"] for row in rows]
    final_ranks = [row["final_video_gt_rank"] for row in rows]
    metrics: Dict[str, float] = {}
    for k in POOLED_TRUE_VR_KS:
        metrics["pooled_true_VR_R@%d" % k] = _recall(pooled_ranks, k)
    for k in LATE_TRUE_VR_KS:
        metrics["late_true_VR_R@%d" % k] = _recall(late_ranks, k)
    for k in FINAL_VIDEO_TRUE_VR_KS:
        metrics["final_video_score_true_VR_R@%d" % k] = _recall(final_ranks, k)

    broad_hit_count = sum(bool(row["broad_hit_at_1000"]) for row in rows)
    late_hit_count = sum(bool(row["late_hit_at_200"]) for row in rows)
    lost_count = sum(
        bool(row["broad_hit_at_1000"]) and not bool(row["late_hit_at_200"])
        for row in rows
    )
    query_count = len(rows)
    g_broad = broad_hit_count / float(query_count)
    g_late = late_hit_count / float(query_count)
    g_keep = late_hit_count / float(broad_hit_count) if broad_hit_count else None
    g_front = _recall(late_ranks, 100)
    l_lost = lost_count / float(broad_hit_count) if broad_hit_count else None
    product_residual = abs(g_late - (g_broad * g_keep)) if g_keep is not None else None
    product_ok = product_residual is not None and product_residual <= G_LATE_PRODUCT_TOLERANCE

    for row in rows:
        row["broad_hit_but_late_lost"] = bool(row["broad_hit_at_1000"]) and not bool(
            row["late_hit_at_200"]
        )

    rank_deltas = [
        float(row["pooled_to_late_rank_delta"])
        for row in rows
        if row["pooled_to_late_rank_delta"] is not None
    ]
    rank_transition = {
        "definition": "pooled_gt_rank_minus_late_gt_rank; positive_is_improvement",
        "available_query_count": len(rank_deltas),
        "missing_query_count": query_count - len(rank_deltas),
        "mean_delta": _mean_or_none(rank_deltas),
        "median_delta": _median_or_none(rank_deltas),
        "improved_query_count": sum(delta > 0.0 for delta in rank_deltas),
        "unchanged_query_count": sum(delta == 0.0 for delta in rank_deltas),
        "degraded_query_count": sum(delta < 0.0 for delta in rank_deltas),
    }
    return {
        "metrics": metrics,
        "gate_metrics": {
            "G_broad": g_broad,
            "G_late": g_late,
            "G_keep": g_keep,
            "G_front": g_front,
            "L_lost": l_lost,
            "G_late_product_residual": product_residual,
            "G_late_product_invariant": product_ok,
        },
        "rank_transition_summary": rank_transition,
        "margin_diagnostics": {
            "pooled": _margin_summary(rows, "pooled"),
            "late": _margin_summary(rows, "late"),
            "final": _margin_summary(rows, "final"),
        },
        "denominators": {
            "query_count": query_count,
            "retrieval_recall_denominator": query_count,
            "conditional_late_retention_denominator": broad_hit_count,
            "rank_transition_denominator": len(rank_deltas),
        },
        "per_query": rows,
    }


def _validate_joint_query(
    query: Mapping[str, Any],
    index: int,
    require_probabilities: bool,
    *,
    input_is_frozen_nms: bool,
) -> Dict[str, Any]:
    try:
        query_id = _require_query_id(query["query_id"], "queries[%d].query_id" % index)
        gt_video = _require_video_id(
            query["gt_video_id"], "queries[%d].gt_video_id" % index
        )
        gt_span = _require_span(query["gt_span_sec"], "queries[%d].gt_span_sec" % index)
        proposals = _normalise_joint_proposals(
            query["joint_proposals"],
            "queries[%d].joint_proposals" % index,
            require_probabilities=require_probabilities,
            preserve_input_order=input_is_frozen_nms,
        )
    except KeyError as error:
        raise MetricContractError(
            "queries[%d] is missing required joint field %s" % (index, error.args[0])
        ) from None
    if "final_video_order" in query:
        final = _normalise_video_ranking(
            query["final_video_order"],
            "queries[%d].final_video_order" % index,
        )
        final_ids = {row.video_id for row in final}
        if any(proposal.video_id not in final_ids for proposal in proposals):
            raise MetricContractError(
                "queries[%d].joint proposal video is outside final_video_order" % index
            )
    return {
        "query_id": query_id,
        "gt_video_id": gt_video,
        "gt_span_sec": gt_span,
        "proposals": proposals,
    }


def _joint_query_metrics(
    row: Mapping[str, Any],
    require_probabilities: bool,
    *,
    input_is_frozen_nms: bool,
) -> Dict[str, Any]:
    query_id = row["query_id"]
    gt_video = row["gt_video_id"]
    gt_span = row["gt_span_sec"]
    proposals: Sequence[_JointProposal] = row["proposals"]
    nms_predictions = list(proposals) if input_is_frozen_nms else _stable_frozen_nms(proposals)
    raw_duplicate_count = len(proposals) - len(
        {
            (proposal.video_id, proposal.start_sec, proposal.end_sec)
            for proposal in proposals
        }
    )

    unique_video_ids: List[str] = []
    seen_videos = set()
    for proposal in nms_predictions:
        if proposal.video_id not in seen_videos:
            seen_videos.add(proposal.video_id)
            unique_video_ids.append(proposal.video_id)

    marginal_rank = None
    false_positive_mass = None
    if require_probabilities and proposals:
        mass_parts_by_video: Dict[str, List[float]] = {}
        for proposal in proposals:
            mass_parts_by_video.setdefault(proposal.video_id, []).append(float(proposal.probability))
        mass_by_video = {
            video_id: math.fsum(parts) for video_id, parts in mass_parts_by_video.items()
        }
        marginal_order = sorted(mass_by_video, key=lambda video: (-mass_by_video[video], video))
        marginal_rank = _rank_in_ids(marginal_order, gt_video)
        false_positive_mass = min(
            1.0,
            max(
                0.0,
                math.fsum(
                    float(proposal.probability)
                    for proposal in proposals
                    if proposal.video_id != gt_video
                    or temporal_iou((proposal.start_sec, proposal.end_sec), gt_span) < 0.3
                ),
            ),
        )

    legacy_rank = _rank_in_ids(unique_video_ids, gt_video)
    top1 = nms_predictions[0] if nms_predictions else None
    top1_available = top1 is not None
    wrong_video = top1 is None or top1.video_id != gt_video
    top1_iou = (
        temporal_iou((top1.start_sec, top1.end_sec), gt_span)
        if top1 is not None and top1.video_id == gt_video
        else 0.0
    )
    correct_video_wrong_span = top1 is not None and top1.video_id == gt_video and top1_iou < 0.5
    top1_false_positive = top1 is None or top1.video_id != gt_video or top1_iou < 0.3

    vc_hits: Dict[str, bool] = {}
    for threshold in VCMR_IOUS:
        for k in VCMR_KS:
            vc_hits["VCMR_R@%d_IoU@%.1f" % (k, threshold)] = any(
                proposal.video_id == gt_video
                and temporal_iou((proposal.start_sec, proposal.end_sec), gt_span) >= threshold
                for proposal in nms_predictions[:k]
            )

    return {
        "query_id": query_id,
        "joint_marginal_gt_rank": marginal_rank,
        "legacy_joint_unique_video_gt_rank": legacy_rank,
        "top1_video_id": top1.video_id if top1 is not None else None,
        "top1_iou": top1_iou,
        "top1_available": top1_available,
        "wrong_video_top1": wrong_video,
        "correct_video_wrong_span_top1": correct_video_wrong_span,
        "top1_false_positive": top1_false_positive,
        "joint_false_positive_mass": false_positive_mass,
        "raw_exact_duplicate_proposal_count": raw_duplicate_count,
        "nms_prediction_count": len(nms_predictions),
        "vcmr_hits": vc_hits,
    }


def evaluate_joint_vcmr_and_errors(
    queries: Sequence[Mapping[str, Any]],
    require_probabilities: bool = True,
    *,
    expected_query_ids: Optional[Sequence[Any]] = None,
    input_is_frozen_nms: bool = False,
) -> Dict[str, Any]:
    """Evaluate joint-marginal, frozen-list VCMR, and disjoint error semantics.

    Normally ``joint_proposals`` are raw normalized ``P_full`` rows.  Marginal
    and false-positive mass use that immutable raw list; VCMR, legacy
    unique-video, and top1 errors use a separately derived frozen per-video NMS
    list.  Historical C7 replay must set ``input_is_frozen_nms=True`` together
    with ``require_probabilities=False`` so the already-NMS artifact is never
    altered or normalized again.
    """

    if not isinstance(queries, (list, tuple)) or not queries:
        raise MetricContractError("queries must be a non-empty explicit sequence")
    if input_is_frozen_nms and require_probabilities:
        raise MetricContractError(
            "frozen post-NMS replay cannot be interpreted as raw normalized P_full"
        )
    rows: List[Dict[str, Any]] = []
    seen_query_ids = set()
    for index, query in enumerate(queries):
        if not isinstance(query, Mapping):
            raise MetricContractError("queries[%d] must be a mapping" % index)
        parsed = _validate_joint_query(
            query,
            index,
            require_probabilities,
            input_is_frozen_nms=input_is_frozen_nms,
        )
        identity = (type(parsed["query_id"]).__name__, parsed["query_id"])
        if identity in seen_query_ids:
            raise MetricContractError("query_id values must be unique with type preserved")
        seen_query_ids.add(identity)
        rows.append(
            _joint_query_metrics(
                parsed,
                require_probabilities,
                input_is_frozen_nms=input_is_frozen_nms,
            )
        )
    _assert_expected_query_universe(
        [row["query_id"] for row in rows], expected_query_ids
    )

    query_count = len(rows)
    joint_metrics: Dict[str, float] = {}
    if require_probabilities:
        marginal_ranks = [row["joint_marginal_gt_rank"] for row in rows]
        for k in JOINT_MARGINAL_VR_KS:
            joint_metrics["joint_marginal_VR_R@%d" % k] = _recall(marginal_ranks, k)
    legacy_ranks = [row["legacy_joint_unique_video_gt_rank"] for row in rows]
    for k in LEGACY_UNIQUE_VIDEO_VR_KS:
        joint_metrics["legacy_joint_unique_video_R@%d" % k] = _recall(legacy_ranks, k)

    vcmr_metrics: Dict[str, float] = {}
    for threshold in VCMR_IOUS:
        for k in VCMR_KS:
            key = "VCMR_R@%d_IoU@%.1f" % (k, threshold)
            vcmr_metrics[key] = sum(bool(row["vcmr_hits"][key]) for row in rows) / float(query_count)
    top1_ious = [float(row["top1_iou"]) for row in rows]
    vcmr_metrics["VCMR_top1_IoU_mean"] = math.fsum(top1_ious) / float(query_count)
    vcmr_metrics["VCMR_top1_IoU_median"] = float(statistics.median(top1_ious))

    mass_values = [
        float(row["joint_false_positive_mass"])
        for row in rows
        if row["joint_false_positive_mass"] is not None
    ]
    errors = {
        "wrong_video_top1_rate": sum(bool(row["wrong_video_top1"]) for row in rows)
        / float(query_count),
        "correct_video_wrong_span_top1_rate": sum(
            bool(row["correct_video_wrong_span_top1"]) for row in rows
        )
        / float(query_count),
        "top1_false_positive_rate": sum(bool(row["top1_false_positive"]) for row in rows)
        / float(query_count),
        "joint_false_positive_mass_mean": _mean_or_none(mass_values),
        "top1_missing_rate": sum(not bool(row["top1_available"]) for row in rows)
        / float(query_count),
        "raw_exact_duplicate_proposal_count": sum(
            int(row["raw_exact_duplicate_proposal_count"]) for row in rows
        ),
        "invalid_span_count": 0,
    }
    for row in rows:
        del row["vcmr_hits"]
        del row["top1_available"]
    return {
        "input_list_semantics": (
            "FROZEN_POST_NMS_NO_REAPPLICATION"
            if input_is_frozen_nms
            else "RAW_P_FULL_WITH_DERIVED_FROZEN_NMS"
        ),
        "nms_reapplied": False if input_is_frozen_nms else True,
        "joint_retrieval_metrics": joint_metrics,
        "vcmr_metrics": vcmr_metrics,
        "error_metrics": errors,
        "denominators": {
            "vcmr_denominator": query_count,
            "top1_error_denominator": query_count,
            "top1_available_count": query_count
            - round(errors["top1_missing_rate"] * query_count),
            "joint_false_positive_mass_denominator": len(mass_values),
        },
        "per_query": rows,
    }


def _high_confidence_metric(
    per_query: Sequence[Mapping[str, Any]],
    prefit_temperature_artifact: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if prefit_temperature_artifact is None:
        return {
            "status": "unavailable",
            "reason": "NO_COMPLIANT_PREFIT_TEMPERATURE",
            "quantile": HIGH_CONFIDENCE_QUANTILE,
            "value": None,
            "fit_performed_during_evaluation": False,
        }
    # F0/F1 has no committed post-hoc calibration capability.  A plain mapping,
    # even one carrying a correct self-hash, is self-signable and therefore can
    # never make a calibration diagnostic available in this window.  A future
    # stage must add a concrete control capability binding the calibration role,
    # checkpoint, policy, evaluator, query universe, parent event, and creation
    # order before this branch may be extended.
    raise MetricContractError(
        "plain prefit temperature mappings are forbidden; a committed control capability is required"
    )


def _unavailable_calibration_diagnostics() -> Dict[str, Any]:
    reason = "NO_COMPLIANT_COMMITTED_PREFIT_TEMPERATURE_CAPABILITY"
    return {
        "ECE@0.5": {
            "status": "unavailable",
            "reason": reason,
            "event": "top1_video_correct_and_IoU_at_least_0.5",
            "candidate_miss_event": 0,
            "value": None,
        },
        "ECE@0.7": {
            "status": "unavailable",
            "reason": reason,
            "event": "top1_video_correct_and_IoU_at_least_0.7",
            "candidate_miss_event": 0,
            "value": None,
        },
        "top1_binary_NLL": {
            "status": "unavailable",
            "reason": reason,
            "all_queries_required": True,
            "value": None,
        },
        "reliability_diagram": {
            "status": "unavailable",
            "reason": reason,
            "binning": "FROZEN_EQUAL_WIDTH_10_BINS",
            "bins": None,
        },
        "structured_joint_NLL_conditional": {
            "status": "unavailable",
            "reason": reason,
            "conditioning": "GT_naturally_in_C_student_and_Y_t_nonempty",
            "value": None,
            "coverage": None,
            "candidate_miss_rate": None,
            "proposal_miss_rate": None,
        },
    }


def evaluate_queries(
    queries: Sequence[Mapping[str, Any]],
    prefit_temperature_artifact: Optional[Mapping[str, Any]] = None,
    *,
    expected_query_ids: Optional[Sequence[Any]] = None,
    require_frozen_capacities: bool = False,
) -> Dict[str, Any]:
    """Evaluate the complete ``c28f_metrics_v1`` contract on explicit rows."""

    if not isinstance(queries, (list, tuple)) or not queries:
        raise MetricContractError("queries must be a non-empty explicit sequence")
    if not isinstance(require_frozen_capacities, bool):
        raise MetricContractError("require_frozen_capacities must be boolean")
    if require_frozen_capacities and expected_query_ids is None:
        raise MetricContractError(
            "real frozen-capacity evaluation requires an exact query manifest"
        )
    for index, query in enumerate(queries):
        if not isinstance(query, Mapping):
            raise MetricContractError("queries[%d] must be a mapping" % index)
        missing = [field for field in REQUIRED_QUERY_FIELDS if field not in query]
        if missing:
            raise MetricContractError(
                "queries[%d] missing required fields: %s" % (index, ",".join(missing))
            )
        if require_frozen_capacities:
            exact_lengths = {
                "pooled_video_order": max(POOLED_TRUE_VR_KS),
                "late_video_order": max(LATE_TRUE_VR_KS),
                "final_video_order": max(FINAL_VIDEO_TRUE_VR_KS),
            }
            for field, expected_length in exact_lengths.items():
                value = query[field]
                if not isinstance(value, (list, tuple)) or len(value) != expected_length:
                    raise MetricContractError(
                        "queries[%d].%s must have frozen length %d"
                        % (index, field, expected_length)
                    )

    retrieval = evaluate_true_video_retrieval(
        queries, expected_query_ids=expected_query_ids
    )
    joint = evaluate_joint_vcmr_and_errors(
        queries,
        require_probabilities=True,
        expected_query_ids=expected_query_ids,
    )
    joint_by_id = {
        (type(row["query_id"]).__name__, row["query_id"]): row for row in joint["per_query"]
    }
    per_query: List[Dict[str, Any]] = []
    for retrieval_row in retrieval["per_query"]:
        identity = (type(retrieval_row["query_id"]).__name__, retrieval_row["query_id"])
        joint_row = joint_by_id.get(identity)
        if joint_row is None:
            raise MetricContractError("retrieval/joint query identity mismatch")
        merged = {
            key: value
            for key, value in retrieval_row.items()
            if not key.endswith("_score_margin") and key != "gt_video_id"
        }
        for key, value in joint_row.items():
            if key == "query_id":
                continue
            if key in merged:
                raise MetricContractError("per-query metric collision: %s" % key)
            merged[key] = value
        per_query.append(merged)

    calibration = _high_confidence_metric(per_query, prefit_temperature_artifact)
    denominators = dict(retrieval["denominators"])
    for key, value in joint["denominators"].items():
        if key in denominators and denominators[key] != value:
            raise MetricContractError("denominator collision: %s" % key)
        denominators[key] = value
    query_count = len(queries)
    return {
        "schema_version": METRIC_SCHEMA_VERSION,
        "unit": UNIT,
        "tie_break": {
            "video_ranking": "score_desc,video_id_asc",
            "unscored_video_ranking": "caller_order_is_frozen_authoritative_order",
            "joint_ranking": "score_desc,video_id_asc,start_sec_asc,end_sec_asc",
            "joint_marginal_ranking": "probability_mass_desc,video_id_asc",
        },
        "nms": {
            "scope": "per_video",
            "iou_operator": ">",
            "iou_threshold": FROZEN_NMS_IOU_THRESHOLD,
            "max_keep": FROZEN_NMS_MAX_KEEP,
            "raw_mass_metrics_run_before_nms": True,
        },
        "retrieval_metrics": retrieval["metrics"],
        "joint_retrieval_metrics": joint["joint_retrieval_metrics"],
        "vcmr_metrics": joint["vcmr_metrics"],
        "error_metrics": joint["error_metrics"],
        "gate_metrics": retrieval["gate_metrics"],
        "rank_transition_summary": retrieval["rank_transition_summary"],
        "margin_diagnostics": retrieval["margin_diagnostics"],
        "calibration": {
            "high_confidence_false_positive_rate": calibration,
            **_unavailable_calibration_diagnostics(),
        },
        "denominators": denominators,
        "completeness": {
            "required_query_count": (
                len(expected_query_ids) if expected_query_ids is not None else query_count
            ),
            "complete_query_count": query_count,
            "missing_required_field_count": 0,
            "ratio": 1.0,
            "query_universe_authority": (
                "EXACT_AUTHORIZED_MANIFEST"
                if expected_query_ids is not None
                else "SELF_CONTAINED_SYNTHETIC_FIXTURE"
            ),
        },
        "diagnostic_availability": {
            "teacher_overlap": {
                "status": "unavailable",
                "reason": "NO_COMMITTED_TEACHER_SHADOW_CAPABILITY",
            },
            "teacher_only_rescue": {
                "status": "unavailable",
                "reason": "NO_COMMITTED_TEACHER_SHADOW_CAPABILITY",
            },
            "gt_support_pre_post_delta": {
                "status": "unavailable",
                "reason": "NO_COMMITTED_GT_INSERTION_COUNTERFACTUAL_CAPABILITY",
            },
        },
        "per_query": per_query,
    }


def evaluate_fixture(fixture: Mapping[str, Any]) -> Dict[str, Any]:
    fixture_map = _require_mapping(fixture, "fixture")
    if fixture_map.get("schema_version") != "c28f_metric_fixture_v1":
        raise MetricContractError("fixture schema_version must be c28f_metric_fixture_v1")
    if fixture_map.get("unit") != UNIT:
        raise MetricContractError("fixture unit must be ratio")
    high_confidence = _require_mapping(fixture_map.get("high_confidence"), "high_confidence")
    if high_confidence.get("status_without_prefit_temperature") != "unavailable":
        raise MetricContractError("fixture must freeze missing-temperature status as unavailable")
    if high_confidence.get("quantile") != HIGH_CONFIDENCE_QUANTILE:
        raise MetricContractError("fixture high-confidence quantile must equal 0.90")
    return evaluate_queries(fixture_map.get("queries"))


def fixture_expected_projection(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the complete hand-checkable projection stored beside the fixture."""

    result_map = _require_mapping(result, "result")
    keys = (
        "schema_version",
        "unit",
        "tie_break",
        "nms",
        "retrieval_metrics",
        "joint_retrieval_metrics",
        "vcmr_metrics",
        "error_metrics",
        "gate_metrics",
        "rank_transition_summary",
        "margin_diagnostics",
        "calibration",
        "diagnostic_availability",
        "denominators",
        "completeness",
        "per_query",
    )
    missing = [key for key in keys if key not in result_map]
    if missing:
        raise MetricContractError("result missing projection keys: %s" % ",".join(missing))
    projection = {key: result_map[key] for key in keys}
    projection["expected_schema_version"] = FIXTURE_EXPECTED_SCHEMA_VERSION
    return projection


def metric_schema() -> Dict[str, Any]:
    """Return the machine-writable metric schema (no filesystem side effect)."""

    return {
        "schema_version": METRIC_SCHEMA_VERSION,
        "unit": UNIT,
        "rank_base": 1,
        "missing_policy": "fail_closed_for_missing_fields; explicit_empty_is_recall_miss",
        "denominator_policy": {
            "recall_and_top1_errors": "all_queries_in_manifest",
            "G_keep_and_L_lost": "queries_with_GT_in_broad1000",
            "joint_false_positive_mass": "queries_with_nonempty_normalized_joint_mass",
            "candidate_miss": "not_skipped",
            "real_evaluation_query_universe": "exact_expected_query_ids_required",
            "self_contained_universe": "synthetic_fixture_only",
        },
        "tie_break": {
            "video_ranking": ["score_desc", "video_id_asc"],
            "unscored_video_ranking": [
                "caller_order_is_frozen_authoritative_order"
            ],
            "joint_ranking": ["score_desc", "video_id_asc", "start_sec_asc", "end_sec_asc"],
            "joint_marginal_ranking": ["probability_mass_desc", "video_id_asc"],
        },
        "nms": {
            "scope": "per_video",
            "iou_operator": ">",
            "iou_threshold": FROZEN_NMS_IOU_THRESHOLD,
            "max_keep": FROZEN_NMS_MAX_KEEP,
            "raw_P_full_preserved_for_marginal_and_false_positive_mass": True,
        },
        "retrieval_families": {
            "pooled_true_VR": {"ks": list(POOLED_TRUE_VR_KS), "proposal_independent": True},
            "late_true_VR": {
                "ks": list(LATE_TRUE_VR_KS),
                "proposal_independent": True,
                "GT_outside_broad": "miss",
            },
            "final_video_score_true_VR": {
                "ks": list(FINAL_VIDEO_TRUE_VR_KS),
                "proposal_independent": True,
            },
            "joint_marginal_VR": {
                "ks": list(JOINT_MARGINAL_VR_KS),
                "definition": "rank_by_sum_of_P_full_over_proposals_per_video",
            },
            "legacy_joint_unique_video": {
                "ks": list(LEGACY_UNIQUE_VIDEO_VR_KS),
                "diagnostic_only": True,
                "definition": "first_occurrence_unique_video_from_frozen_NMS_VCMR_list",
            },
        },
        "vcmr": {"ks": list(VCMR_KS), "iou_thresholds": list(VCMR_IOUS)},
        "errors": {
            "wrong_video_top1": "top1_video_id != GT_video_id",
            "correct_video_wrong_span_top1": "top1_video_is_GT_and_IoU_lt_0.5",
            "top1_false_positive": "no_top1_or_wrong_video_or_correct_video_IoU_lt_0.3",
            "joint_false_positive_mass": "P_full_mass_on_wrong_video_or_correct_video_IoU_lt_0.3",
            "high_confidence_false_positive_rate": {
                "quantile": HIGH_CONFIDENCE_QUANTILE,
                "without_compliant_prefit_temperature": "unavailable",
                "fit_during_evaluation": "forbidden",
                "plain_self_hashed_mapping_authority": "forbidden",
            },
        },
        "calibration_diagnostics": {
            "ECE@0.5": {
                "event": "top1_video_correct_and_IoU_at_least_0.5",
                "all_queries_including_candidate_miss": True,
            },
            "ECE@0.7": {
                "event": "top1_video_correct_and_IoU_at_least_0.7",
                "all_queries_including_candidate_miss": True,
            },
            "top1_binary_NLL": {
                "all_queries_including_candidate_miss": True,
            },
            "reliability_diagram": {"binning": "FROZEN_EQUAL_WIDTH_10_BINS"},
            "structured_joint_NLL_conditional": {
                "conditioning": "GT_naturally_in_C_student_and_Y_t_nonempty",
                "required_companion_rates": [
                    "coverage",
                    "candidate_miss_rate",
                    "proposal_miss_rate",
                ],
            },
            "current_f0_f1_availability": (
                "UNAVAILABLE_UNTIL_CONCRETE_COMMITTED_CALIBRATION_CAPABILITY"
            ),
        },
        "counterfactual_diagnostics": {
            "teacher_overlap": "requires_committed_teacher_shadow_capability",
            "teacher_only_rescue": "requires_committed_teacher_shadow_capability",
            "gt_support_pre_post_delta": (
                "requires_committed_gt_insertion_counterfactual_capability"
            ),
            "missing_capability_status": "unavailable",
        },
        "legacy_fields": {"overwrite": "forbidden", "new_fields_only": True},
    }


def gate_schema() -> Dict[str, Any]:
    """Return the machine-writable G2 gate registry contribution."""

    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "unit": UNIT,
        "missing_policy": "fail_closed",
        "thresholds": {
            "G_broad": {"metric": "pooled_true_VR_R@1000", "operator": ">=", "value": G_BROAD_THRESHOLD},
            "G_keep": {"metric": "conditional_late_retention", "operator": ">=", "value": G_KEEP_THRESHOLD},
            "G_front": {"metric": "late_true_VR_R@100", "operator": ">=", "value": G_FRONT_THRESHOLD},
            "query_completeness": {"operator": "==", "value": QUERY_COMPLETENESS_THRESHOLD},
            "historical_vcmr_parity_max_abs_delta": {"operator": "<=", "value": HISTORICAL_PARITY_TOLERANCE},
        },
        "invariants": {
            "G_late_product": {
                "equation": "G_late == G_broad * G_keep",
                "absolute_tolerance": G_LATE_PRODUCT_TOLERANCE,
                "G_late_is_independent_evidence": False,
            }
        },
    }


def evaluate_gates(
    metric_result: Mapping[str, Any],
    parity_max_abs_delta: float,
    *,
    require_exact_authorized_manifest: bool = False,
) -> Dict[str, Any]:
    result = _require_mapping(metric_result, "metric_result")
    if not isinstance(require_exact_authorized_manifest, bool):
        raise MetricContractError("require_exact_authorized_manifest must be boolean")
    if result.get("schema_version") != METRIC_SCHEMA_VERSION:
        raise MetricContractError("gate input must use c28f_metrics_v1")
    gates = _require_mapping(result.get("gate_metrics"), "metric_result.gate_metrics")
    retrieval = _require_mapping(
        result.get("retrieval_metrics"), "metric_result.retrieval_metrics"
    )
    completeness = _require_mapping(result.get("completeness"), "metric_result.completeness")
    required = ("G_broad", "G_late", "G_keep", "G_front", "G_late_product_invariant")
    missing = [field for field in required if field not in gates]
    if missing:
        raise MetricContractError("gate metrics missing: %s" % ",".join(missing))
    for field in ("G_broad", "G_late", "G_keep", "G_front"):
        if not _is_real_number(gates[field]):
            raise MetricContractError("%s is unavailable; gates fail closed" % field)
        if not 0.0 <= float(gates[field]) <= 1.0:
            raise MetricContractError("%s is outside ratio unit bounds" % field)
    frozen_bindings = {
        "G_broad": retrieval.get("pooled_true_VR_R@1000"),
        "G_late": retrieval.get("late_true_VR_R@200"),
        "G_front": retrieval.get("late_true_VR_R@100"),
    }
    for field, source_value in frozen_bindings.items():
        if not _is_real_number(source_value) or not math.isclose(
            float(gates[field]),
            float(source_value),
            rel_tol=0.0,
            abs_tol=G_LATE_PRODUCT_TOLERANCE,
        ):
            raise MetricContractError("%s is not bound to its frozen retrieval metric" % field)
    if float(gates["G_broad"]) == 0.0:
        raise MetricContractError(
            "G_keep is undefined when the frozen broad-hit denominator is zero"
        )
    expected_keep = float(gates["G_late"]) / float(gates["G_broad"])
    if not math.isclose(
        float(gates["G_keep"]),
        expected_keep,
        rel_tol=0.0,
        abs_tol=G_LATE_PRODUCT_TOLERANCE,
    ):
        raise MetricContractError("G_keep is not the frozen conditional retention")
    expected_residual = abs(
        float(gates["G_late"])
        - float(gates["G_broad"]) * float(gates["G_keep"])
    )
    if not _is_real_number(gates.get("G_late_product_residual")) or not math.isclose(
        float(gates["G_late_product_residual"]),
        expected_residual,
        rel_tol=0.0,
        abs_tol=G_LATE_PRODUCT_TOLERANCE,
    ):
        raise MetricContractError("G_late product residual is inconsistent")
    if not isinstance(gates["G_late_product_invariant"], bool) or gates[
        "G_late_product_invariant"
    ] != (
        expected_residual <= G_LATE_PRODUCT_TOLERANCE
    ):
        raise MetricContractError("G_late product invariant is inconsistent")
    if not _is_real_number(completeness.get("ratio")):
        raise MetricContractError("query completeness is unavailable")
    if require_exact_authorized_manifest and completeness.get(
        "query_universe_authority"
    ) != "EXACT_AUTHORIZED_MANIFEST":
        raise MetricContractError("real gate evaluation lacks exact query authority")
    if type(completeness.get("required_query_count")) is not int or type(
        completeness.get("complete_query_count")
    ) is not int:
        raise MetricContractError("query completeness counts are invalid")
    if completeness["required_query_count"] <= 0 or not 0 <= completeness[
        "complete_query_count"
    ] <= completeness["required_query_count"]:
        raise MetricContractError("query completeness count bounds are invalid")
    expected_ratio = completeness["complete_query_count"] / float(
        completeness["required_query_count"]
    )
    if not math.isclose(
        float(completeness["ratio"]), expected_ratio, rel_tol=0.0, abs_tol=0.0
    ):
        raise MetricContractError("query completeness ratio/count mismatch")
    if not _is_real_number(parity_max_abs_delta) or float(parity_max_abs_delta) < 0.0:
        raise MetricContractError("parity_max_abs_delta must be finite and non-negative")
    decisions = {
        "G_broad": float(gates["G_broad"]) >= G_BROAD_THRESHOLD,
        "G_keep": float(gates["G_keep"]) >= G_KEEP_THRESHOLD,
        "G_front": float(gates["G_front"]) >= G_FRONT_THRESHOLD,
        "G_late_product": gates["G_late_product_invariant"] is True,
        "query_completeness": float(completeness["ratio"]) == QUERY_COMPLETENESS_THRESHOLD,
        "historical_vcmr_parity": float(parity_max_abs_delta) <= HISTORICAL_PARITY_TOLERANCE,
    }
    return {"decisions": decisions, "all_pass": all(decisions.values())}


def additive_merge_no_overwrite(
    legacy_fields: Mapping[str, Any], c28f_fields: Mapping[str, Any]
) -> Dict[str, Any]:
    """Add C28F fields without ever overwriting a legacy top-level field."""

    legacy = _require_mapping(legacy_fields, "legacy_fields")
    current = _require_mapping(c28f_fields, "c28f_fields")
    collisions = sorted(set(legacy).intersection(current))
    if collisions:
        raise MetricContractError("legacy field overwrite forbidden: %s" % ",".join(collisions))
    merged = dict(legacy)
    merged.update(current)
    return merged


def compare_vcmr_parity(
    legacy_metrics: Mapping[str, Any],
    current_metrics: Mapping[str, Any],
    field_map: Optional[Mapping[str, str]] = None,
    tolerance: float = HISTORICAL_PARITY_TOLERANCE,
    legacy_unit: str = UNIT,
    current_unit: str = UNIT,
) -> Dict[str, Any]:
    """Compare frozen legacy VCMR fields in ratio units.

    C7 legacy reports may store percentages while C28F stores ratios.  The caller
    must state both units explicitly when they differ; conversion is deterministic
    and the frozen tolerance is always applied in ratio units.
    """

    legacy = _require_mapping(legacy_metrics, "legacy_metrics")
    current = _require_mapping(current_metrics, "current_metrics")
    if not _is_real_number(tolerance) or float(tolerance) < 0.0:
        raise MetricContractError("parity tolerance must be finite and non-negative")
    if legacy_unit not in ("ratio", "percent") or current_unit not in ("ratio", "percent"):
        raise MetricContractError("parity units must be ratio or percent")
    mapping = dict(field_map) if field_map is not None else {
        "VCMR_R@%d_IoU@%.1f" % (k, threshold): "VCMR_R@%d_IoU@%.1f" % (k, threshold)
        for threshold in VCMR_IOUS
        for k in VCMR_KS
    }
    if not mapping:
        raise MetricContractError("parity field map must not be empty")
    deltas: Dict[str, float] = {}
    for legacy_key, current_key in mapping.items():
        if legacy_key not in legacy or current_key not in current:
            raise MetricContractError("parity field missing: %s -> %s" % (legacy_key, current_key))
        if not _is_real_number(legacy[legacy_key]) or not _is_real_number(current[current_key]):
            raise MetricContractError("parity fields must be finite numeric values")
        legacy_value = float(legacy[legacy_key]) / (100.0 if legacy_unit == "percent" else 1.0)
        current_value = float(current[current_key]) / (100.0 if current_unit == "percent" else 1.0)
        if not 0.0 <= legacy_value <= 1.0 or not 0.0 <= current_value <= 1.0:
            raise MetricContractError("normalized parity recall must be in [0,1]")
        deltas[legacy_key] = abs(legacy_value - current_value)
    max_delta = max(deltas.values())
    return {
        "status": "PASS" if max_delta <= float(tolerance) else "FAIL",
        "tolerance": float(tolerance),
        "comparison_unit": UNIT,
        "legacy_input_unit": legacy_unit,
        "current_input_unit": current_unit,
        "max_abs_delta": max_delta,
        "field_abs_deltas": deltas,
        "compared_field_count": len(deltas),
    }


def iter_aligned_c7_replay_rows(
    prediction_stream: Iterable[Mapping[str, Any]],
    score_stream: Iterable[Mapping[str, Any]],
    *,
    video_id_decoder: Callable[[int], str],
    span_decoder: Callable[[Any, int, int, int], Tuple[float, float]],
) -> Iterator[Dict[str, Any]]:
    """Purely align streams already opened by the concrete G2 control handle.

    This function has deliberately no authority Mapping, path, factory, or open
    API.  It cannot grant access.  The G2 stage must persist and consume its
    one-shot control capability before obtaining these iterables.
    """

    if not isinstance(prediction_stream, Iterable) or not isinstance(score_stream, Iterable):
        raise MetricContractError("C7 replay streams must be iterable")
    if not callable(video_id_decoder) or not callable(span_decoder):
        raise MetricContractError("C7 decoders must be callable")

    sentinel = object()
    current_query = sentinel
    current_rows: List[List[Any]] = []
    completed_queries = set()
    previous_rank: Optional[int] = None
    pair_count = 0

    for pair_index, (prediction, score_row) in enumerate(
        zip_longest(prediction_stream, score_stream, fillvalue=sentinel)
    ):
        if prediction is sentinel or score_row is sentinel:
            raise MetricContractError("C7 prediction/score streams have different lengths")
        pair_count += 1
        prediction_map = _require_mapping(prediction, "prediction_stream[%d]" % pair_index)
        score_map = _require_mapping(score_row, "score_stream[%d]" % pair_index)
        for field in ("q", "rank"):
            if field not in prediction_map or field not in score_map:
                raise MetricContractError("C7 aligned rows require q and rank")
            if prediction_map[field] != score_map[field]:
                raise MetricContractError("C7 prediction/score q-rank alignment mismatch")
        query_id = _require_query_id(prediction_map["q"], "prediction.q")
        rank = prediction_map["rank"]
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
            raise MetricContractError("C7 rank must be a non-negative integer")
        video_index = prediction_map.get("video_idx")
        start_index = prediction_map.get("start_idx")
        end_index = prediction_map.get("end_idx")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in (video_index, start_index, end_index)):
            raise MetricContractError("C7 video/start/end indices must be non-negative integers")
        score = score_map.get("score")
        if not _is_real_number(score):
            raise MetricContractError("C7 score must be finite")

        if current_query is sentinel or query_id != current_query:
            if current_query is not sentinel:
                completed_queries.add((type(current_query).__name__, current_query))
                yield {"query_id": current_query, "joint_proposals": current_rows}
            identity = (type(query_id).__name__, query_id)
            if identity in completed_queries:
                raise MetricContractError("C7 query rows must be contiguous")
            current_query = query_id
            current_rows = []
            previous_rank = None
        expected_rank = 0 if previous_rank is None else previous_rank + 1
        if rank != expected_rank:
            raise MetricContractError("C7 ranks must be contiguous from zero within a query")
        if rank >= FROZEN_NMS_MAX_KEEP:
            raise MetricContractError("C7 rank exceeds the frozen post-NMS cap")
        previous_rank = rank

        video_id = _require_video_id(video_id_decoder(video_index), "decoded_video_id")
        decoded_span = _require_span(
            span_decoder(query_id, video_index, start_index, end_index), "decoded_span"
        )
        current_rows.append([video_id, decoded_span[0], decoded_span[1], float(score)])

    if pair_count == 0:
        raise MetricContractError("C7 replay streams must not be empty")
    yield {"query_id": current_query, "joint_proposals": current_rows}


__all__ = [
    "C7_LEGACY_VCMR_PARITY_FIELD_MAP",
    "FIXTURE_EXPECTED_SCHEMA_VERSION",
    "GATE_SCHEMA_VERSION",
    "HISTORICAL_PARITY_TOLERANCE",
    "METRIC_SCHEMA_VERSION",
    "MetricContractError",
    "additive_merge_no_overwrite",
    "compare_vcmr_parity",
    "evaluate_fixture",
    "evaluate_gates",
    "evaluate_joint_vcmr_and_errors",
    "evaluate_queries",
    "evaluate_true_video_retrieval",
    "fixture_expected_projection",
    "gate_schema",
    "iter_aligned_c7_replay_rows",
    "metric_schema",
    "temporal_iou",
]
