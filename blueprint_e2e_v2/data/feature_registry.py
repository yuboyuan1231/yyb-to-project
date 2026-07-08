from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blueprint_e2e_v2.utils.hashing import file_sha256, stable_hash


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/home/a/yybwork/data/yyb/tvr_feature_release")
TMP_ROOT = Path("/tmp/c28c_score_cache/CONQUER-RLEM-c2c3")


@dataclass(frozen=True)
class FeaturePaths:
    data_root: Path = DATA_ROOT
    train_jsonl: Path = DATA_ROOT / "data/tvr_train_select100_release.jsonl"
    train_full_jsonl: Path = DATA_ROOT / "data/tvr_train_release.jsonl"
    val_jsonl: Path = DATA_ROOT / "data/tvr_val_release.jsonl"
    video_meta: Path = DATA_ROOT / "data/tvr_video2dur_idx.json"
    first_stage_rank_lmdb: Path = DATA_ROOT / "data/train_select100_top2000"
    query_lmdb: Path = DATA_ROOT / "sub_query_feature/tvr_query_pretrained_w_sub_query"
    subtitle_lmdb: Path = DATA_ROOT / "sub_query_feature/tvr_sub_pretrained_w_sub_query_max_cl-1.5"
    visual_lmdb: Path = DATA_ROOT / "video_feature/resnet_slowfast_1.5"
    split_dir: Path = REPO_ROOT / "c12_1_schema_and_split/splits"
    c26r_reference_multispan: Path = Path("/tmp/c26r_score_cache/CONQUER-RLEM-c2c3/C26R_C17_C26_JOINED_MULTI_SPAN_medium_seed2026.local.parquet")
    c17_reference_score_table: Path = Path("/tmp/c17_score_cache/CONQUER-RLEM-c2c3/C17_2_SCORE_TABLE_medium.local.parquet")


class FeatureRegistry:
    def __init__(self, paths: FeaturePaths | None = None) -> None:
        self.paths = paths or FeaturePaths()

    def required_files(self) -> dict[str, Path]:
        return {
            "train_jsonl": self.paths.train_jsonl,
            "video_meta": self.paths.video_meta,
            "query_lmdb_data": self.paths.query_lmdb / "data.mdb",
            "subtitle_lmdb_data": self.paths.subtitle_lmdb / "data.mdb",
            "visual_lmdb_data": self.paths.visual_lmdb / "data.mdb",
            "train_fit_split": self.paths.split_dir / "train_fit_desc_ids.txt",
            "calib_select_split": self.paths.split_dir / "calib_select_desc_ids.txt",
            "calib_holdout_split": self.paths.split_dir / "calib_holdout_desc_ids.txt",
        }

    def optional_reference_files(self) -> dict[str, Path]:
        return {
            "first_stage_rank_lmdb_data": self.paths.first_stage_rank_lmdb / "data.mdb",
            "c26r_reference_multispan": self.paths.c26r_reference_multispan,
            "c17_reference_score_table": self.paths.c17_reference_score_table,
        }

    def dependency_audit(self) -> dict[str, Any]:
        modules = ["lmdb", "msgpack", "msgpack_numpy", "numpy", "torch", "pandas"]
        return {m: importlib.util.find_spec(m) is not None for m in modules}

    def audit(self) -> dict[str, Any]:
        required = {
            name: {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
                "sha256": file_sha256(path) if path.exists() and path.is_file() and path.stat().st_size < 64 * 1024 * 1024 else None,
            }
            for name, path in self.required_files().items()
        }
        optional = {
            name: {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
            }
            for name, path in self.optional_reference_files().items()
        }
        deps = self.dependency_audit()
        missing_required = [k for k, v in required.items() if not v["exists"]]
        missing_deps = [k for k, v in deps.items() if not v]
        return {
            "data_root": str(self.paths.data_root),
            "tmp_root": str(TMP_ROOT),
            "required": required,
            "optional_reference_only": optional,
            "dependencies": deps,
            "missing_required": missing_required,
            "missing_dependencies": missing_deps,
            "registry_hash": stable_hash({"required": required, "optional": optional, "deps": deps}),
        }

