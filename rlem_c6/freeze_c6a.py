#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rlem_c6.c6a_utils import artifact, freeze_gates, write_json, write_text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dirs", nargs="+", required=True)
    p.add_argument("--audit_dir", default="c6_audit")
    args = p.parse_args()
    records = []
    for d in args.run_dirs:
        path = Path(d) / "best_epoch.json"
        if path.exists():
            rec = json.loads(path.read_text(encoding="utf-8"))
            rec["run_dir"] = d
            rec["freeze_gates"] = freeze_gates(rec)
            records.append(rec)
    if not records:
        raise ValueError("no best_epoch.json found")
    passed = [r for r in records if r["freeze_gates"]["core_gate"]]
    selected = max(passed or records, key=lambda r: r.get("selection_score", -1e9))
    status = "C6A_FREEZE_REVIEW_PASS" if passed else "C6A_NEGATIVE"
    ad = Path(args.audit_dir)
    ad.mkdir(parents=True, exist_ok=True)
    stem = "FREEZE" if passed else "NEGATIVE"
    review = ad / ("C6A_FREEZE_REVIEW.md" if passed else "C6A_NEGATIVE_AUDIT.md")
    manifest_path = ad / f"C6A_{stem}_MANIFEST.json"
    hashes_path = ad / f"C6A_{stem}_HASHES.json"
    manifest = {
        "stage": "C6-A",
        "status": status,
        "selected_config": Path(selected["run_dir"]).name,
        "selected_epoch": selected["epoch"],
        "selected_checkpoint": str(Path(selected["run_dir"]) / "model_best.pt"),
        "selected_metrics": selected["metrics"],
        "selected_deltas_vs_c4_final": selected.get("deltas_vs_c4_final", {}),
        "selected_movement": selected.get("movement", {}),
        "selected_localization": selected.get("localization", {}),
        "selected_freeze_gates": selected["freeze_gates"],
        "all_runs": [
            {
                "run_dir": r["run_dir"],
                "epoch": r["epoch"],
                "selection_score": r["selection_score"],
                "gates": r["freeze_gates"],
            }
            for r in records
        ],
        "official_val_used": False,
        "post_val_adjustment": False,
        "backbone_finetuned": False,
        "candidate_generation_modified": False,
        "NMS_modified": False,
        "evaluator_modified": False,
        "C6_B_used": False,
        "C6_C_used": False,
    }
    write_json(manifest_path, manifest)
    md = f"# C6-A {'freeze review' if passed else 'negative audit'}\n\n"
    md += f"- Status: `{status}`\n"
    md += f"- Selected run: `{Path(selected['run_dir']).name}`\n"
    md += f"- Selected epoch: `{selected['epoch']}`\n"
    md += "- Official val used: `false`\n"
    md += "- Post-val adjustment: `false`\n\n"
    md += "## Deltas vs C4_final\n\n"
    md += "".join(f"- `{k}`: `{v}`\n" for k, v in selected.get("deltas_vs_c4_final", {}).items())
    md += "\n## Freeze gates\n\n"
    md += "".join(f"- `{k}`: `{v}`\n" for k, v in selected["freeze_gates"].items())
    write_text(review, md)
    hashes = {
        "status": "HASHES_MATERIALIZED",
        "official_val_used": False,
        "artifacts": {
            "review": artifact(review),
            "manifest": artifact(manifest_path),
        },
    }
    for r in records:
        d = Path(r["run_dir"])
        prefix = d.name
        hashes["artifacts"][f"{prefix}_model_best"] = artifact(d / "model_best.pt")
        hashes["artifacts"][f"{prefix}_history"] = artifact(d / "history.json")
        hashes["artifacts"][f"{prefix}_scores"] = artifact(d / "train_calib_scores_best.npz")
    write_json(hashes_path, hashes)
    print(json.dumps({"status": status, "selected": Path(selected["run_dir"]).name}, indent=2))


if __name__ == "__main__":
    main()
