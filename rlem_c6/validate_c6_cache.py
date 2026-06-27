#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rlem_c6.c6a_utils import C6AdapterModel, C6GroupDataset, artifact, c6_collate, write_json, write_text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_cache", required=True)
    p.add_argument("--train_features", required=True)
    p.add_argument("--train_temporal", required=True)
    p.add_argument("--calib_cache", required=True)
    p.add_argument("--calib_features", required=True)
    p.add_argument("--calib_temporal", required=True)
    p.add_argument("--audit_md", required=True)
    p.add_argument("--audit_json", required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = p.parse_args()
    checks = {}
    status = "PASS"
    for label, temporal_path in [("train_fit", args.train_temporal), ("train_calib", args.calib_temporal)]:
        with np.load(temporal_path, allow_pickle=False) as tp:
            for name in ["p_b", "p_e", "p_ctx"]:
                arr = tp[name]
                checks[f"{label}_{name}_finite"] = bool(np.isfinite(arr).all())
                checks[f"{label}_{name}_nonnegative"] = bool((arr >= 0).all())
                checks[f"{label}_{name}_sum_max_abs_error"] = float(np.max(np.abs(arr.sum(axis=1) - 1.0)))
                if not checks[f"{label}_{name}_finite"] or not checks[f"{label}_{name}_nonnegative"] or checks[f"{label}_{name}_sum_max_abs_error"] > 1e-3:
                    status = "FAIL"
    ds = C6GroupDataset(args.train_cache, args.train_features, args.train_temporal, max_groups=8)
    batch = c6_collate([ds[i] for i in range(min(4, len(ds)))])
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = C6AdapterModel(
        row_feat_dim=batch["row_feat"].shape[-1],
        video_feat_dim=batch["video_feat"].shape[-1],
        hidden_dim=128,
        temporal_layers=2,
        heads=4,
        ffn_dim=256,
        span_head_hidden=192,
        span_head_layers=2,
    ).to(device)
    rows = batch.pop("rows")
    with torch.no_grad():
        out = model({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}, zero_adapter=True)
    diff = float(torch.max(torch.abs(out["score"] - batch["s_c4"].to(device))).item())
    checks["zero_adapter_max_abs_diff"] = diff
    checks["gpu_forward_device"] = str(device)
    checks["batch_groups"] = int(out["score"].shape[0])
    if diff > 1e-6:
        status = "FAIL"
    audit = {
        "status": status,
        "checks": checks,
        "official_val_used": False,
        "artifacts": {
            "train_cache": artifact(args.train_cache),
            "train_calib_cache": artifact(args.calib_cache),
        },
    }
    write_json(args.audit_json, audit)
    md = "# C6-A cache audit\n\n"
    md += f"- Status: `{status}`\n"
    md += f"- GPU forward device: `{device}`\n"
    md += f"- Zero-adapter max abs diff: `{diff}`\n"
    md += "- Official val used: `false`\n\n"
    md += "\n".join(f"- `{k}`: `{v}`" for k, v in checks.items()) + "\n"
    write_text(args.audit_md, md)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
