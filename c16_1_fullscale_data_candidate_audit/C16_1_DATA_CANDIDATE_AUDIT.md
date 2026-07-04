# C16-1 Data / Candidate Audit

status = `C16_FULLSCALE_DATA_READY`

- mode: `medium`
- train_fit queries: `1000`
- calib_select queries: `1000`
- calib_holdout queries: `1000`
- pseudo_official_holdout queries: `1000`
- rebuilt_pool: `true`
- feature_schema_hash: `fcedf9a713a56ec334c819fc129d3d1bbb7a269f79ce90f9371c6a990ef5a75f`
- candidate_pool_hash: `ccd364258d055362a17e9ad3d37540e667753ac8e830c2d2b41883096e442ed5`

The candidate pool is a native C16 dense start-duration map, not C7-B6 fixed
prediction rows. pseudo_official_holdout is audited but not used for selection.

official was not run.
