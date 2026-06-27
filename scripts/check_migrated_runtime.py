#!/usr/bin/env python3

"""Check paths and frozen artifacts after moving the CONQUER-RLEM workspace."""

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path


REQUIRED_DATA_KEYS = [
    "desc_bert_path", "sub_bert_path", "vid_feat_path", "train_data_path",
    "eval_data_path", "test_data_path", "train_first_VR_ranklist_path",
    "eval_first_VR_ranklist_path_hero", "test_public_first_VR_ranklist_path_hero",
    "video_duration_idx_path",
]
FROZEN_ARTIFACTS = {
    "results/rlem_c31_capacity/best_config.json": "ed9ac0f5a30596d1c6fbbf87cd5dbf017edd5ad9ed0a869c2ae899a1841b6eb8",
    "c31_audit/C31_FROZEN_MANIFEST.json": "530f4466b3bc80d75fd246127f01de0274c8eecb3a941a494d5fc749b98d1408",
    "results/rlem_c31_official_val/VAL_C31_ONE_SHOT_MANIFEST.json": "a3a46899ee2411cd0da63e0d015330c80375375a9bf54114cc55b6413ba87599",
    "results/rlem_c31_capacity/models/cap_F_wd0/model_best.pt": "eb3559ff35853a7702b34a06cea1da66de594315b1bc40c89aabcd713ea55c42",
}
EXPECTED_PACKAGES = {
    "torch": ("torch", "2.0.1+cu117"),
    "numpy": ("numpy", "1.26.4"),
    "scipy": ("scipy", "1.10.1"),
    "sklearn": ("scikit-learn", "1.6.1"),
    "tqdm": ("tqdm", "4.65.2"),
    "easydict": ("easydict", "1.13"),
    "lmdb": ("lmdb", "1.5.1"),
    "msgpack": ("msgpack", "1.1.2"),
    "msgpack_numpy": ("msgpack-numpy", "0.4.8"),
    "tensorboard": ("tensorboard", "2.20.0"),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--dataset_config", default="config/tvr_data_config.json")
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    config_path = Path(args.dataset_config)
    if not config_path.is_absolute():
        config_path = project / config_path
    config = json.load(open(config_path, encoding="utf-8"))
    dataset_root = Path(config["root_path"])
    path_checks = {}
    for key in REQUIRED_DATA_KEYS:
        value = Path(config[key])
        resolved = value if value.is_absolute() else dataset_root / value
        path_checks[key] = {"path": str(resolved), "exists": resolved.exists()}
    artifact_checks = {}
    for relative, expected in FROZEN_ARTIFACTS.items():
        path = project / relative
        actual = sha256_file(path) if path.exists() else None
        artifact_checks[relative] = {
            "exists": path.exists(), "expected_sha256": expected,
            "actual_sha256": actual, "matches": actual == expected,
        }
    dependencies = {}
    for module, (distribution, expected) in EXPECTED_PACKAGES.items():
        installed = importlib.util.find_spec(module) is not None
        try:
            actual = importlib.metadata.version(distribution) if installed else None
        except importlib.metadata.PackageNotFoundError:
            actual = None
        dependencies[module] = {
            "installed": installed, "distribution": distribution,
            "expected": expected, "actual": actual, "matches": actual == expected,
        }
    torch_runtime = None
    if dependencies["torch"]["installed"]:
        torch = importlib.import_module("torch")
        torch_runtime = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": torch.version.cuda,
            "cuda_runtime_expected": "11.7",
            "cuda_runtime_matches": torch.version.cuda == "11.7",
            "cudnn": torch.backends.cudnn.version(),
            "cudnn_expected_major_minor": "8.5",
            "cudnn_matches": str(torch.backends.cudnn.version()).startswith("85"),
        }
    path_ok = dataset_root.is_dir() and all(item["exists"] for item in path_checks.values())
    artifacts_ok = all(item["matches"] for item in artifact_checks.values())
    runtime_ok = (
        sys.version.startswith("3.9.19")
        and all(item["matches"] for item in dependencies.values())
        and torch_runtime is not None
        and torch_runtime["cuda_available"]
        and torch_runtime["cuda_runtime_matches"]
        and torch_runtime["cudnn_matches"]
    )
    result = {
        "status": "PASS" if path_ok and artifacts_ok and runtime_ok else "BLOCKED",
        "project_root": str(project), "dataset_root": str(dataset_root),
        "dataset_paths_ok": path_ok, "frozen_artifacts_ok": artifacts_ok,
        "python": sys.executable, "python_version": sys.version.split()[0],
        "dependencies": dependencies, "torch_runtime": torch_runtime,
        "runtime_dependencies_ok": runtime_ok,
        "path_checks": path_checks, "artifact_checks": artifact_checks,
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
