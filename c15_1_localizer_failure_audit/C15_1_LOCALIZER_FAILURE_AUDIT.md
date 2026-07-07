# C15-1 Localizer Failure Audit

status = `C15_FAILURE_SCORE_MAP_REQUIRED`

- same-sample query count: `300`
- T2 IoU@0.7 top100: `52.6667`
- T2 generated oracle IoU@0.7: `62.3333`
- T2 lost good top100 rate: `9.6667`
- T2 PQ/IoU Spearman: `-0.13312608862140654`
- raw start peak close within 2 clips: `24.0000`
- raw end peak close within 2 clips: `18.3333`

Conclusion: C15 should test a BMN-style query-aware dense span confidence map
instead of continuing row-level text/event score tuning.

official was not run.
