# C4 Full Audit Requirements

## Current allowed stage by default

Only C4-lite train_calib is authorized unless the user explicitly authorizes later stages.

## Required C4-lite report

- official_val_used=false
- candidate rows/query = 200
- effective_top_n=100
- NMS=0.7
- max_after_nms=100
- C3.1 frozen artifacts unchanged
- no R2/VS head
- no CONQUER modification
- grid results and best config
- deltas versus C3.1 train_calib reference
- movement/safety diagnostics

## Required C4-r2-cal report

Only after explicit authorization:

- train_fit/train_calib video dataset manifests
- feature standardizer fit on train_fit only
- external residual MLP checkpoint and diagnostics
- train_calib-only grid search
- no official val
- no CONQUER source modification

## Required C4-main report

Only after separate authorization and accepted source-specific protocol:

- source diff
- checkpoint compatibility report
- new baseline reproduction
- training/eval protocol
- explicit statement that C0/C1/C2/C3.1 frozen artifacts remain historical and are not overwritten
