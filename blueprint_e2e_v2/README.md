# C28C CONQUER-RLEM-E2E v2.0

Clean-room release-feature end-to-end VCMR blueprint.

This package does not continue the C26S static integration route. It reads
TVR/HERO release features, builds trainable retrieval and localization modules,
refreshes dynamic candidates from the current retriever, keeps multi-span
proposal materialization, and reports train-only metrics without official
validation or official prediction pools.

Large local artifacts are written under:

`/tmp/c28c_score_cache/CONQUER-RLEM-c2c3`

