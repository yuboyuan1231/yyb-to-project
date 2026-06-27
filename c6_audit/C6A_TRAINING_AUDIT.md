# C6-A training audit

- Status: `C6A_TRAINING_COMPLETE_NO_FREEZE_GATE_PASS`
- Official val used: `false`
- Post-val adjustment: `false`
- Backbone finetuned: `false`
- Candidate generation modified: `false`

## Config comparison

### c6a_default

- Epochs: `8`
- Best epoch: `1`
- Best selection score: `-0.09178270251464862`
- Deltas vs C4_final:
  - `0.5-r1`: `0.011442956860051368`
  - `0.5-r10`: `-0.034328870580168314`
  - `0.5-r100`: `0.022885913720102735`
  - `0.5-r5`: `0.0`
  - `0.7-r1`: `-0.034328870580157655`
  - `0.7-r10`: `-0.04577182744020547`
  - `0.7-r100`: `0.0343288705801541`
  - `0.7-r5`: `-0.022885913720102735`
- Movement:
  - `hard_positive_top100_entries`: `55`
  - `hard_positive_top100_exit_ratio`: `0.0004780952684504932`
  - `hard_positive_top100_exits`: `30`
  - `mean_within_query_spearman`: `0.999864080536102`
  - `pearson_base_candidate`: `0.9999802603917503`
  - `top1_changed_ratio`: `0.005607048861425793`
- Localization:
  - `best_iou_span_rank_delta_mean`: `0.003123668891097544`
  - `oracle_video_r1_05_delta`: `-0.05679397983813317`
  - `oracle_video_r1_07_delta`: `-0.04259548487860343`
  - `selected_span_miou_delta`: `-0.0003810017257950271`

### c6a_large

- Epochs: `8`
- Best epoch: `1`
- Best selection score: `-0.11458096543554076`
- Deltas vs C4_final:
  - `0.5-r1`: `0.0`
  - `0.5-r10`: `-0.04577182744021968`
  - `0.5-r100`: `0.011442956860051368`
  - `0.5-r5`: `0.011442956860051368`
  - `0.7-r1`: `0.0`
  - `0.7-r10`: `-0.09154365488041805`
  - `0.7-r100`: `0.0`
  - `0.7-r5`: `-0.05721478430026394`
- Movement:
  - `hard_positive_top100_entries`: `34`
  - `hard_positive_top100_exit_ratio`: `0.0003027936700186457`
  - `hard_positive_top100_exits`: `19`
  - `mean_within_query_spearman`: `0.9999164848671509`
  - `pearson_base_candidate`: `0.9999910926622376`
  - `top1_changed_ratio`: `0.005149330587023687`
- Localization:
  - `best_iou_span_rank_delta_mean`: `0.004259548487860287`
  - `oracle_video_r1_05_delta`: `-0.02839698991907369`
  - `oracle_video_r1_07_delta`: `0.0`
  - `selected_span_miou_delta`: `-0.0004164886126781786`

### c6a_regularized

- Epochs: `8`
- Best epoch: `1`
- Best selection score: `-0.06315140560108524`
- Deltas vs C4_final:
  - `0.5-r1`: `-0.011442956860051368`
  - `0.5-r10`: `0.0`
  - `0.5-r100`: `0.04577182744021968`
  - `0.5-r5`: `0.045771827440212576`
  - `0.7-r1`: `-0.011442956860051368`
  - `0.7-r10`: `-0.04577182744020547`
  - `0.7-r100`: `0.011442956860051368`
  - `0.7-r5`: `-0.045771827440212576`
- Movement:
  - `hard_positive_top100_entries`: `32`
  - `hard_positive_top100_exit_ratio`: `0.0004302857416054439`
  - `hard_positive_top100_exits`: `27`
  - `mean_within_query_spearman`: `0.9998969126305055`
  - `pearson_base_candidate`: `0.9999862057277454`
  - `top1_changed_ratio`: `0.0037761757638173706`
- Localization:
  - `best_iou_span_rank_delta_mean`: `0.004117563538264944`
  - `oracle_video_r1_05_delta`: `-0.01419849495952974`
  - `oracle_video_r1_07_delta`: `0.0`
  - `selected_span_miou_delta`: `-0.00010202773656931008`

