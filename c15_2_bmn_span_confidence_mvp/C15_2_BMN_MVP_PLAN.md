# C15-2 BMN-style Span Confidence MVP

This MVP implements a query-aware start-duration span confidence map over
existing clip-level features. It does not use raw video and does not rerank a
fixed candidate row pool as the model's core operation.

Variants:

- B1_boundary_only_map
- B2_boundary_actionness_map
- B3_query_aware_map
- B4_duration_conditioned_map
- B5_short_specialized_map

Training uses train_fit, selection uses calib_select, and reporting uses
calib_holdout. pseudo_official_holdout and official validation are not used.
