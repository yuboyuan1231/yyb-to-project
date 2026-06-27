#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rlem_c6.c6a_utils import artifact, write_json, write_text


def main() -> None:
    ad = Path("c6_audit")
    ad.mkdir(parents=True, exist_ok=True)
    state = {
        "stage": "C6-A adapter-only trainable mutualization",
        "c4_final": "C4-r2-cal-v2.1 v21_00444",
        "c5_fixed_candidate_status": "official-val negative / not promoted",
        "official_val_used_for_c6_selection": False,
        "c6_protocol_new_stage": True,
        "backbone_full_finetune_allowed": False,
        "candidate_regeneration_allowed": False,
        "official_val_allowed_now": False,
    }
    write_json(ad / "C6A_START_STATE.json", state)
    write_text(
        ad / "C6A_START_STATE.md",
        "# C6-A start state\n\n" + "".join(f"- `{k}`: `{v}`\n" for k, v in state.items()),
    )
    protocol = """# C6-A protocol

C6-A is an adapter-only trainable stage. It keeps `C4_final = C4-r2-cal-v2.1 v21_00444` frozen, does not modify the CONQUER backbone, does not regenerate candidates, and does not modify NMS or the evaluator.

Primary baseline: frozen C4_final. Secondary references: C4-lite, C3.1, and C5-main-A3b-wide official-val negative.

This phase trains only a lightweight-but-nontrivial adapter/head on train_fit and selects on train_calib. It validates whether the retrieval→localization temporal-prior signal that fixed-candidate C5 could not stably consume can be learned by a trainable branch.

Official val is not authorized in this phase. Phase 7 stops at freeze review or negative audit. Any official-val one-shot requires separate user authorization.
"""
    write_text(ad / "C6A_PROTOCOL.md", protocol)
    hashes = {
        "status": "HASHES_MATERIALIZED",
        "official_val_used": False,
        "artifacts": {
            "start_state_json": artifact(ad / "C6A_START_STATE.json"),
            "start_state_md": artifact(ad / "C6A_START_STATE.md"),
            "protocol": artifact(ad / "C6A_PROTOCOL.md"),
        },
    }
    write_json(ad / "C6A_START_STATE_HASHES.json", hashes)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
