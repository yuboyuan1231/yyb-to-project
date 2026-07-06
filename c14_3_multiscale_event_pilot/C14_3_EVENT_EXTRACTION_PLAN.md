# C14-3 Event Extraction Plan

This event pilot does not depend on raw video. It uses existing visual energy,
subtitle/query similarity, and generated C12 M1000 sample pools.

Implemented event families:

1. fixed-window event pooling: windows 2, 4, 8, 16 clips.
2. adjacent-similarity event grouping: deterministic 1D adjacent visual-energy
   merge, selected threshold 0.75.
3. time-contiguous segment clustering proxy: 4 ordered quantile segments.

Config hash: `cd885cba79ce1684de532487c0e633fdb1190cef752c981dd72fabf9eb7ee50c`

official was not run.
