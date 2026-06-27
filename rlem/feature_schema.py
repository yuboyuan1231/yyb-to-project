
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

"""Feature schema for CONQUER-RLEM C2/C3.

The C2/C3 heads deliberately use compact scalar evidence exported by
``rlem/export_conquer_evidence.py``.  Missing fields are allowed because some
CONQUER checkpoints do not enable all optional branches; missing values are
imputed as 0 and normalized with the stored feature statistics.
"""

from typing import Iterable, List


DEFAULT_C3_FEATURES: List[str] = [
    # retrieval evidence
    "r1",
    "r1_tilde",
    "rank_r1",
    "r2_raw",
    "r2_prob",
    "r2_tilde",
    "rank_r2",
    "r_abs_gap",
    # localization evidence
    "b_start_logit",
    "e_end_logit",
    "p_b_i",
    "p_e_j",
    "l_logit",
    "l_prob",
    "l_prod",
    # boundary evidence
    "u_b",
    "u_e",
    "u_bd",
    "sharp_b",
    "sharp_e",
    "margin_b",
    "margin_e",
    # modality evidence
    "mu_v",
    "mu_s",
    "b_mod",
    "h_mod",
    # QAL / context temporal evidence
    "m_ctx",
    "mean_ctx",
    "max_ctx",
    "peak_in_ctx",
    "h_ctx",
    # duration / position evidence
    "span_len",
    "span_len_norm",
    "start_norm",
    "end_norm",
    # base score, useful as a calibration feature but not the final score itself
    "s_base",
]

R2_FEATURES = ["r2_raw", "r2_prob", "r2_tilde", "rank_r2", "r_abs_gap"]
DEFAULT_C2_NO_R2_FEATURES: List[str] = [
    name for name in DEFAULT_C3_FEATURES if name not in R2_FEATURES
]


QSP_FEATURES: List[str] = [
    "qsp_mass_v", "qsp_mean_v", "qsp_max_v", "qsp_peak_in_v", "qsp_entropy_v",
    "qsp_mass_s", "qsp_mean_s", "qsp_max_s", "qsp_peak_in_s", "qsp_entropy_s",
    "qsp_mass_gate", "qsp_mean_gate", "qsp_max_gate", "qsp_peak_in_gate", "qsp_entropy_gate",
    "qsp_js_vs", "qsp_gate_confidence", "qsp_sub_coverage",
    "qsp_q_v", "qsp_q_s", "qsp_modality_gap", "qsp_modality_entropy",
]

DEFAULT_C35_QSP_FEATURES: List[str] = DEFAULT_C2_NO_R2_FEATURES + QSP_FEATURES
C35_QSP_DROP_COVERAGE_FEATURES: List[str] = [
    name for name in DEFAULT_C35_QSP_FEATURES if name != "qsp_sub_coverage"
]
C35_MATCHED_FEATURE_SETS = {
    "control31": list(DEFAULT_C2_NO_R2_FEATURES),
    "qsp52_drop_coverage": list(C35_QSP_DROP_COVERAGE_FEATURES),
    "qsp53_with_coverage": list(DEFAULT_C35_QSP_FEATURES),
}


LABEL_FIELDS: List[str] = [
    "y_joint",
    "y_joint_05",
    "y_joint_07",
    "m_bd",
    "y_bd",
    "y_fp",
]


def parse_feature_list(text: str) -> List[str]:
    """Parse a comma-separated feature list or return the default schema."""
    if text and text.strip().lower() in {"c2", "c2_no_r2", "no_r2"}:
        return list(DEFAULT_C2_NO_R2_FEATURES)
    if text and text.strip().lower() in {"c35", "c35_qsp", "qsp"}:
        return list(DEFAULT_C35_QSP_FEATURES)
    if not text or text.strip().lower() in {"default", "c3"}:
        return list(DEFAULT_C3_FEATURES)
    return [x.strip() for x in text.split(",") if x.strip()]


def compact_feature_string(features: Iterable[str]) -> str:
    return ",".join(features)
