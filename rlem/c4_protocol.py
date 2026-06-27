#!/usr/bin/env python
"""Protocol guards for CONQUER-RLEM C4 full scaffold.

The code is intentionally staged.  C4-lite and C4-r2-cal are implemented as
external fixed-candidate calibration.  C4-main/VS-head integration is guarded and
must not be trained until separately authorized.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

C4_ALLOWED_STAGES = {
    "compile_only",
    "c4_lite_score_candidates",
    "c4_lite_build_cache",
    "c4_lite_search",
    "c4_lite_one_shot",
    "c4_r2_build_video_dataset",
    "c4_r2_train_calibrator",
    "c4_r2_grid_search",
    "c4_r2_one_shot",
    "c4_main_dry_run",
}

FROZEN_EFFECTIVE_TOP_N = 100
FROZEN_MAX_AFTER_NMS = 100
FROZEN_NMS_THD = 0.7

FORBIDDEN_FEATURES_R2 = {"r2_raw", "r2_prob", "r2_tilde", "rank_r2", "r_abs_gap"}


@dataclass(frozen=True)
class C4Protocol:
    stage: str
    split: str
    official_val_allowed: bool = False
    r2_main_allowed: bool = False
    effective_top_n: int = FROZEN_EFFECTIVE_TOP_N
    max_after_nms: int = FROZEN_MAX_AFTER_NMS
    nms_thd: float = FROZEN_NMS_THD
    r2_features_enabled: bool = False
    modifies_conquer: bool = False
    reads_official_val: bool = False

    def validate(self) -> None:
        if self.stage not in C4_ALLOWED_STAGES:
            raise ValueError(f"Unknown C4 stage: {self.stage}")
        if self.effective_top_n != FROZEN_EFFECTIVE_TOP_N:
            raise ValueError("C4 requires effective_top_n=100")
        if self.max_after_nms != FROZEN_MAX_AFTER_NMS:
            raise ValueError("C4 requires max_after_nms=100")
        if not math.isclose(float(self.nms_thd), FROZEN_NMS_THD, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("C4 requires nms_thd=0.7")
        if self.split == "val" and not self.official_val_allowed:
            raise ValueError("Official val is refused without explicit one-shot authorization")
        if self.reads_official_val and not self.official_val_allowed:
            raise ValueError("reads_official_val=true requires official_val_allowed=true")
        if self.r2_features_enabled:
            raise ValueError("Legacy r2 feature columns remain forbidden in C4-lite/r2-cal")
        if self.modifies_conquer and not self.r2_main_allowed:
            raise ValueError("CONQUER source modification requires explicit C4-main authorization")

    def to_manifest(self) -> Dict:
        obj = asdict(self)
        obj["protocol_validated"] = True
        obj["frozen_effective_top_n"] = FROZEN_EFFECTIVE_TOP_N
        obj["frozen_max_after_nms"] = FROZEN_MAX_AFTER_NMS
        obj["frozen_nms_thd"] = FROZEN_NMS_THD
        return obj


def validate_no_forbidden_r2_features(feature_names: Iterable[str]) -> None:
    bad = sorted(FORBIDDEN_FEATURES_R2 & set(feature_names))
    if bad:
        raise ValueError(f"Forbidden unavailable r2 features enabled: {bad}")


def ensure_output_namespace(path: str, allowed_prefix: str = "results/rlem_c4") -> None:
    p = Path(path)
    # Only enforce for relative project outputs.  Absolute /tmp paths are allowed
    # for tests, but normal project outputs must use the C4 namespace.
    if not p.is_absolute() and not str(p).startswith(allowed_prefix):
        raise ValueError(f"C4 output must be under {allowed_prefix}: {path}")


def write_protocol_manifest(path: str, protocol: C4Protocol, extra: Optional[Dict] = None) -> None:
    obj = protocol.to_manifest()
    if extra:
        obj.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


__all__ = [
    "C4Protocol", "validate_no_forbidden_r2_features", "ensure_output_namespace",
    "write_protocol_manifest", "FROZEN_EFFECTIVE_TOP_N", "FROZEN_MAX_AFTER_NMS",
    "FROZEN_NMS_THD", "FORBIDDEN_FEATURES_R2",
]
