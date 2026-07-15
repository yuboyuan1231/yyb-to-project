# C28F v5 F0/F1 key code and conclusions

This handoff is the compact GitHub entry point for the complete C28F v5 F0/F1 implementation. It covers the scientific contracts, fail-closed authority and persistence controls, the executable runtime path, tests, and the frozen F0/F1 conclusions. Large runtime traces and data-derived per-query outputs are intentionally excluded from the publication branch.

## Code map

The complete implementation is under `blueprint_e2e_v2/c28f_v5/`.

- Scientific and evaluation contracts: `a4_metrics.py`, `a4_stages.py`, `a4_roles.py`, `a4_f1.py`, `a4_forward.py`, and `a4_finalize.py`.
- ID-only projection and safety controls: `a4_id_projector.py`, `a4_security.py`, `atomic_io.py`, `canonical.py`, `constants.py`, `control_plane.py`, and `a4_control.py`.
- Durable F0/F1 execution and recovery: `runtime_store.py`, `runtime_control.py`, `runtime_data.py`, `runtime_workflow.py`, `runtime_migration.py`, `runtime_stages.py`, and `runtime_runner.py`.
- Entrypoints: `run_c28f_v5_f0_f1.py` and `a4_runner.py`.
- Tests and frozen fixtures: `tests/c28f_v5/`.

The active recovery runner exposes exactly these six business actions:

1. `F0_A_ANALYZE`
2. `G4_ROLE_POLICY_LOCK`
3. `RUN_F0_B`
4. `F0_B_ANALYZE`
5. `G6_VERIFY_F1`
6. `G7_FINALIZE`

It has no F0-A rerun, real-data training, protected evaluation, official evaluation, or F2+ execution action.

## Frozen F0/F1 conclusions

- Terminal state: `F0_F1_WINDOW_COMPLETE_STOPPED`, transition sequence `1386`, state SHA256 `1349d1de50683222524c59a16c4924419be4c0c9fcf0831be12aea25a5b93361`.
- G4 used all `69,428` frozen train-fit IDs. The sorted unique manifest SHA256 is `24007d3f6f1159f55e68a8ee64c779436a34db4bff601f9d9cd74ee6e102bf0b`.
- The five role counts are calibration `2,049`, route-dev `2,051`, mechanism-dev `16,406`, confirm `16,404`, and core `32,518`. Their desc-ID and GT-video sets are pairwise disjoint and their desc-ID union is exactly `69,428`.
- F0-A retained its original single successful logical evaluation: eval `F0A-EVAL-8b80aa50b2b215a1235e8ab2ed3ac49b`, run `F0A-RUNTIME-71a02539b3dab25d8fe3ff0bea1a71ec`.
- F0-B used exactly one successful logical evaluation: eval `F0B-EVAL-ac14798e9d2fc9a33786a7cc187ed37a`, run `F0B-RUNTIME-2ecf3f7e40aa55cf4aadd05d54e316ba`, covering `2,051` route-dev queries in `257` committed chunks.
- Both frozen evaluations first fail at `POOLED_BROAD`; the final routing result is `ROUTING_CONSISTENT`.
- G6 completed the synthetic F1 protocol with `64` synthetic optimizer updates, `0` real-data optimizer updates, and no extra model-forward split. F2-F5 scientific mechanisms remain `NOT_APPLICABLE_SCHEMA_ONLY`.
- Across the committed runtime lineage, model forward count is exactly two: one F0-A and one F0-B. Other/protected model forwards, real-data optimizer updates, protected evaluation, holdout/official evaluation, and F2+ execution are all zero.
- The recovery lineage has zero forbidden protected-data access. The separately recorded pre-A4 exception remains excluded from the recovery scientific lineage, with scientific impact `NONE` and protocol-audit impact `MATERIAL_RECORDED_EXCEPTION`.
- `NEXT_AUTHORIZATION_REQUEST.json` is a request only. It grants no current authority, does not start F2, and cannot continue automatically.

## Published conclusion evidence

- Implementation review: `blueprint_e2e_v2/reports/c28f_v5/goal_runtime/artifacts/review/IMPLEMENTATION_CONSISTENCY_REPORT.md`
- Test report: `blueprint_e2e_v2/reports/c28f_v5/goal_runtime/artifacts/review/TEST_EXECUTION_REPORT.json`
- G3 F0-A analysis: `blueprint_e2e_v2/reports/c28f_v5/goal_runtime/artifacts/g3/f0_forensics/f0_a_forensics_report.json`
- G4 role/policy lock: `blueprint_e2e_v2/reports/c28f_v5/goal_runtime/artifacts/g4/f1_protocol/role_manifest_lock.json`
- G5 routing: `blueprint_e2e_v2/reports/c28f_v5/goal_runtime/artifacts/g5/f0_forensics/routing_decision.json`
- G6 F1 protocol: `blueprint_e2e_v2/reports/c28f_v5/goal_runtime/artifacts/g6/f1_protocol/f1_protocol_report.final.json`
- G7 final decision: `blueprint_e2e_v2/reports/c28f_v5/goal_runtime/artifacts/g7/f1_protocol/finalization/F0_F1_WINDOW_DECISION.json`
- Protected-data accounting: `blueprint_e2e_v2/reports/c28f_v5/goal_runtime/artifacts/g7/f1_protocol/finalization/protected_data_access_summary.json`

The final artifact-manifest file SHA256 is `5c43097b18d03d5f903453dfa9df75fa279dd3f13d930308b07656dc07518ca4`.

## Validation

- Targeted runtime recovery tests: `9 passed`.
- Full `tests/c28f_v5` suite: `347 passed`.
- No-data temporary runtime state-path dry run: `PASS`.
- Complete terminal event/state/transaction chain: `1,386` entries validated.
- G3/G4/G5/G6/G7 committed output counts: `9 / 15 / 7 / 8 / 9`.

## Deliberate publication exclusions

This branch does not publish raw datasets, feature stores, checkpoints, model weights, caches, per-query metrics, rank-transition parquet files, forward chunks, heartbeats, transaction journals, event/state mirrors, or unrelated historical experiment directories. Those are execution material rather than key F0/F1 source or conclusions.
