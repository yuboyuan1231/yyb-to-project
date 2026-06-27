# C5-main-inference-A2B negative audit

- Status: `NO_PROMOTION`
- Mode: `video_anchor`
- Selected: `c5ma_0000`
- Best nonzero: `c5ma_0098`
- Official val used: `false`

## Best nonzero deltas vs C4_final
- `0.5-r1`: `0.045771827440212576`
- `0.5-r5`: `0.42338940382194323`
- `0.5-r10`: `0.37761757638173776`
- `0.7-r1`: `0.12587252546057925`
- `0.7-r5`: `0.011442956860051368`
- `0.7-r10`: `-0.17164435290078472`
- `0.5-r100`: `0.011442956860051368`
- `0.7-r100`: `-0.2174161803410044`

## Localization
- oracle R1@0.5 delta: `0.01419849495952974`
- oracle R1@0.7 delta: `0.31236688910975374`
- selected mIoU delta: `0.000644958134141338`
- best-IoU mean-rank delta: `-0.028255004969473235`

## Safety
- hard-positive exit ratio: `0.0176417154058232`
- anchor drift max abs: `0.0`
- video top-10 membership changed ratio: `0.0`

Execution stopped before official val.
