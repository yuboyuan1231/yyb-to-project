from __future__ import annotations

from pathlib import Path


SCHEMA_VERSION = "c28f_v5_f0_f1_control_v1"
GOAL_ID = "019f50ee-e80f-7b30-a0f7-1f1a2808b8ea"
GOAL_OBJECTIVE = (
    "/home/a/yybwork/yybcounter/plan/再改进2/CODEX_C28F_V5_F0_F1_GOAL.md"
    "按照指令执行，编码后运行代码前先对代码审核确认无误再开始。"
    "优先GPU0(如果有进程直接kill掉),cpu尽量多核"
)

REPO_ROOT = Path("/home/a/yybwork/yybcounter/CONQUER-RLEM-c2c3")
GOAL_CONTRACT = Path("/home/a/yybwork/yybcounter/plan/再改进2/CODEX_C28F_V5_F0_F1_GOAL.md")
BLUEPRINT = Path("/home/a/yybwork/yybcounter/plan/再改进2/C28F_CMGC_v5.0_证据门控实验蓝图.md")
HANDOFF = REPO_ROOT / "handoff/C28E3_NEXT_WINDOW_HANDOFF_20260711.md"
CHECKPOINT = Path(
    "/home/a/yybwork/yybcounter/c28c_score_cache/CONQUER-RLEM-c2c3/"
    "checkpoints/C28C_FULL_medium_seed2026.pt"
)

EXPECTED_BRANCH = "c28e-code-review-clip-late-interaction"
EXPECTED_HEAD = "3ee00135cc00c9763d616fa1b49c64206742a007"
EXPECTED_GOAL_CONTRACT_SHA256 = "1d0269b6cfcce3f6f917039df14c00c66fd23e3e800e09d50db1b1b06a856c73"
EXPECTED_BLUEPRINT_SHA256 = "dfab46788e7f2c5efc82eaa9440afbc5fd55d608a849636af7fe9303f9b6799d"
EXPECTED_HANDOFF_SHA256 = "c3d2759de354cf199b86ac3aac5ae62f5a772225a0ea6ca9a3db043536a75d82"
EXPECTED_CHECKPOINT_SHA256 = "fa0bafc6b5c1a0d049d462bcd0b77a073222715af0713ae291c05ea8631d13ab"
EXPECTED_CHECKPOINT_SIZE = 97_704_933

CODE_ROOT = REPO_ROOT / "blueprint_e2e_v2/c28f_v5"
TEST_ROOT = REPO_ROOT / "tests/c28f_v5"
ENTRYPOINT = REPO_ROOT / "run_c28f_v5_f0_f1.py"
REPORT_ROOT = REPO_ROOT / "blueprint_e2e_v2/reports/c28f_v5"
CACHE_ROOT = REPO_ROOT / "c28f_v5_cache"
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints/c28f_v5"
CONTROL_ROOT = REPORT_ROOT / "goal_control"
QUARANTINE_ROOT = REPORT_ROOT / "quarantine"
BOOTSTRAP_LOCK = REPORT_ROOT / "bootstrap.lock"
BOOTSTRAP_LOCK_HEADER = b"C28F_BOOTSTRAP_RECOVERY_LOCK_V1\n"

# The first committed G0 control transaction was rejected by its own final
# verifier because one list depended on pre-serialization dict insertion
# order.  Its bytes are preserved outside the canonical control root and are
# locked here as non-scientific, quarantined failure evidence.
QUARANTINED_G0_ATTEMPTS = {
    "G0-A2-dcd3b04dde5d4607b48cf5431d707114": {
        "transaction_id": "G0-TXN-7960a7ecd1dc480f8b000f8952437458",
        "control_tree_sha256": "8d48ce583a8690ea5a789bd500f53d5fd85b0d178c9610f5a1394c2d2041d44e",
        "file_count": 29,
        "directory_count": 5,
        "root_cause": "DICT_INSERTION_ORDER_DIVERGED_AFTER_CANONICAL_JSON_RELOAD",
        "scientific_impact": "NONE",
        "model_forward_evaluations": 0,
        "protected_content_opens_current_attempt": 0,
        "failure_stage": "FINAL_EXACT_NAMESPACE_VERIFY",
        "observed_state_sha256": "64202fd4223a9e70ad91210112cf22eceae1dce90af648385145c1761a1050e7",
        "observed_event_sha256": "44b2a85587093cc443d77e017123386e658708a59d1cc310d4d99274e479c129",
        "observed_budget_file_sha256": "692a9082f70e6349a5b6c41b21604429ac0ad574bb18e68b39129846b6578751",
        "intent_file_sha256": "6d6f77bd967ce9e42eb150a14a14ed36b66bda8b83e2172d5c55ef856d1c1b12",
        "static_review_receipt_sha256": "dc515a0db412f713e6607586836f856efd270a2220c36c6a9beb6b466f844aa9",
        "failure_fingerprint": {
            "extra_event_record_sha256": "44b2a85587093cc443d77e017123386e658708a59d1cc310d4d99274e479c129",
            "missing_event_record_sha256": "6984195dc1ced192b5b9ad5f4b7b6df80107f574706c9bae14959bbfdd4d76b1",
            "extra_budget_file_sha256": "692a9082f70e6349a5b6c41b21604429ac0ad574bb18e68b39129846b6578751",
            "missing_budget_file_sha256": "04e314047ac3504d2d37f3046a38466a2970cca4327438cd594c6cfc043526ea",
        },
    }
}

CANONICAL_ROOTS = {
    "code": CODE_ROOT,
    "tests": TEST_ROOT,
    "entrypoint": ENTRYPOINT,
    "reports": REPORT_ROOT,
    "cache": CACHE_ROOT,
    "checkpoint": CHECKPOINT_ROOT,
    "state": CONTROL_ROOT,
}

GOAL_OWNED_REPO_PREFIXES = (
    "blueprint_e2e_v2/c28f_v5/",
    "tests/c28f_v5/",
    "blueprint_e2e_v2/reports/c28f_v5/",
    "c28f_v5_cache/",
    "checkpoints/c28f_v5/",
)
GOAL_OWNED_EXACT_REPO_PATHS = ("run_c28f_v5_f0_f1.py",)

ALLOWED_WRITE_GLOBS = (
    str(CODE_ROOT / "**"),
    str(TEST_ROOT / "**"),
    str(ENTRYPOINT),
    str(REPORT_ROOT / "**"),
    str(CACHE_ROOT / "**"),
    str(CHECKPOINT_ROOT / "**"),
)

HISTORICAL_DENY_WRITE_ROOTS = (
    Path("/home/a/yybwork/yybcounter/c28c_score_cache/CONQUER-RLEM-c2c3"),
    Path("/tmp/c28c_score_cache/CONQUER-RLEM-c2c3"),
    REPO_ROOT / "blueprint_e2e_v2/reports/c28e_3_training",
    REPO_ROOT / "blueprint_e2e_v2/reports/c28c_0_protocol",
    REPO_ROOT / "blueprint_e2e_v2/reports/c28c_1_data",
    REPO_ROOT / "blueprint_e2e_v2/reports/c28c_2_training",
    REPO_ROOT / "blueprint_e2e_v2/reports/c28c_3_eval",
    REPO_ROOT / "blueprint_e2e_v2/reports/c28c_4_ablation",
    REPO_ROOT / "blueprint_e2e_v2/reports/c28c_5_robustness",
    REPO_ROOT / "blueprint_e2e_v2/reports/c28c_6_final",
)

DATA_ROOT = Path("/home/a/yybwork/data/yyb/tvr_feature_release")
TRAIN_JSONL = DATA_ROOT / "data/tvr_train_select100_release.jsonl"
VIDEO_META = DATA_ROOT / "data/tvr_video2dur_idx.json"
PROTECTED_VAL_JSONL = DATA_ROOT / "data/tvr_val_release.jsonl"
PROTECTED_HISTORICAL_C28C_RESULTS = (
    REPO_ROOT / "blueprint_e2e_v2/reports/c28c_3_eval/C28C_3_VCMR_RESULTS.json"
)
PRE_G0_SECURITY_INCIDENT = {
    "event_id": "PRE_G0_PATH_CLASSIFICATION_001",
    "operations": [
        "primary rg split-marker path classification",
        "delegated static reviewer sed head inspection",
        "delegated static reviewer rg full-file scan",
    ],
    "policy_status_at_detection": "SECURITY_INCIDENT_AWAITING_USER_ACK",
    "current_disposition": "ACKNOWLEDGED_CONTINUE_WITH_RECORDED_EXCEPTION",
    "content_stream_open_count_at_least": 3,
    "fields_surfaced": [
        "path",
        "split",
        "query_id",
        "gt_video_id",
        "gt_start",
        "gt_end",
        "video_id",
        "span_start",
        "span_end",
        "rank_after_nms",
        "vcmr_score",
        "video_score",
    ],
    "prediction_values_surfaced": "NONZERO_UNQUANTIFIED",
    "model_score_values_surfaced": "NONZERO_UNQUANTIFIED",
    "label_or_timestamp_values_surfaced": "NONZERO_UNQUANTIFIED",
    "metric_values_surfaced": 0,
    "query_text_surfaced": 0,
    "model_evaluation_count": 0,
    "derived_scientific_artifact_count": 0,
    "quarantine_scope": "NO_SCIENTIFIC_DESCENDANTS_BOOTSTRAP_UNEXECUTED",
    "content_hash_computed": False,
    "lineage_action": "artifact removed from baseline and replay allowlists",
}
PRE_G0_SECURITY_INCIDENT_USER_ACK = True
PRE_G0_SECURITY_INCIDENT_USER_ACK_TEXT = "先评估影响，如果无影响，就继续"
PRE_G0_SECURITY_INCIDENT_IMPACT_ASSESSMENT = {
    "scientific_decision_impact": "NONE",
    "protocol_audit_impact": "MATERIAL_RECORDED_EXCEPTION",
    "reasons": [
        "no model forward or evaluator invocation occurred",
        "neither F0-A nor F0-B eval token was reserved or consumed",
        "no scientific prediction, metric, threshold, route, or model artifact was produced",
        "the delegated reviewer that viewed protected fields was stopped and will not be reused",
        "the protected artifact is permanently excluded from baseline and replay lineage",
    ],
    "continuation_scope": (
        "new G0 attempt only; no protected content access is authorized except a separately journaled "
        "ID-only mapping operation under AUTH_PROTECTED_ID_MAPPING_ONLY"
    ),
}
SPLIT_ROOT = REPO_ROOT / "c12_1_schema_and_split/splits"
TRAIN_FIT_MANIFEST = SPLIT_ROOT / "train_fit_desc_ids.txt"
CALIB_SELECT_MANIFEST = SPLIT_ROOT / "calib_select_desc_ids.txt"
CALIB_HOLDOUT_MANIFEST = SPLIT_ROOT / "calib_holdout_desc_ids.txt"
PSEUDO_OFFICIAL_MANIFEST = SPLIT_ROOT / "pseudo_official_holdout_desc_ids.txt"

FEATURE_SOURCES = {
    "query_lmdb": DATA_ROOT / "sub_query_feature/tvr_query_pretrained_w_sub_query/data.mdb",
    "subtitle_lmdb": DATA_ROOT / "sub_query_feature/tvr_sub_pretrained_w_sub_query_max_cl-1.5/data.mdb",
    "visual_lmdb": DATA_ROOT / "video_feature/resnet_slowfast_1.5/data.mdb",
}

TRAIN_CORPUS_COUNT = 17_435
TRAIN_CORPUS_VIDEO_ID_ORDER_SHA256 = "51ebc37675f9d822fdd5ce1d92608bc7f442893d89451e2eb301f2d3d640676e"
TRAIN_CORPUS_AUDITS = {
    "c12_feature_inventory": {
        "path": REPO_ROOT / "c12_3_feature_audit/C12_3A_FEATURE_INVENTORY.md",
        "sha256": "57475e5a966c5ccae33453b2528f35a995464d0b3c7e6547293ae8d605e06fe1",
    },
    "c12_corpus_index_audit": {
        "path": REPO_ROOT / "c12_4_native_retriever/C12_4A_CORPUS_INDEX_AUDIT.json",
        "sha256": "a3c4e4814f1e2ee133e7ce42a68d6bb07a941366a33ec3ce1f7a89ddd32ee8ce",
    },
}

HISTORICAL_ARTIFACTS = {
    "c28e_decision": REPORT_ROOT.parent / "c28e_3_training/C28E_3_FULL_E2E_TRAINING_DECISION.json",
    "c28e_training_log": REPORT_ROOT.parent / "c28e_3_training/C28C_FULL_medium_seed2026.training_log.json",
    "c28e_stream_log": REPORT_ROOT.parent / "c28e_3_training/C28E3_formal_cuda0_20260709_stream.log",
    "c28e_config": REPO_ROOT / "blueprint_e2e_v2/configs/c28e_code_review.yaml",
}

BOOTSTRAP_FIXTURES = {
    "metric_fixture_v1": TEST_ROOT / "fixtures/metric_fixture_v1.json",
    "state_fixture_v1": TEST_ROOT / "fixtures/state_fixture_v1.json",
}

STATIC_REVIEW_RECEIPT = TEST_ROOT / "STATIC_REVIEW_RECEIPT.json"
G0_REVIEWED_FILES = (
    CODE_ROOT / "__init__.py",
    CODE_ROOT / "constants.py",
    CODE_ROOT / "canonical.py",
    CODE_ROOT / "atomic_io.py",
    CODE_ROOT / "control_plane.py",
    ENTRYPOINT,
    BOOTSTRAP_FIXTURES["metric_fixture_v1"],
    BOOTSTRAP_FIXTURES["state_fixture_v1"],
)
