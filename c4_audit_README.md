# C4-lite Audit Requirements

Required checks before any official-val C4-lite evaluation:

- official_val_used=false during all search;
- candidate rows/query remain 200;
- effective_top_n=100, NMS=0.7, max_after_nms=100;
- no R2/VS head is trained or read;
- C3.1 frozen artifacts remain unchanged;
- search uses train_calib only;
- best config records aggregation means from train_calib only;
- official val requires a separate one-shot authorization and must use exactly
  the frozen best config.

Recommended pass criteria:

- most primary R@1/R@5/R@10 deltas over the C3.1 train-calib reference are positive;
- R@100 deltas are not materially negative;
- base score remains a major component;
- top-1 changed ratio is not an abnormal pool rewrite;
- hard-positive top-100 exits remain low.
